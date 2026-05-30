# This is the final pipeline
import cv2
import time
import threading
import os
import subprocess
import queue
from Steering_wheel_detector import detect_steering_and_hands, draw_landmarks_on_image, draw_pose_markers
from Feedback import generate_safety_alert_all_groq, get_ffmpeg_cmd
from action_recognition import ActionRecognizer

def get_wav_duration(audio_bytes: bytes) -> float:
    """Return duration of WAV audio bytes in seconds.
    
    If it's not a valid WAV or fails to parse, fallback to 2.0s.
    """
    import wave
    import io
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except Exception:
        return 2.0

# Fixed variables
SAME_ACTION_COOLDOWN = 5.0      # seconds
DIFF_ACTION_COOLDOWN = 2.0      # seconds

HANDS_OFF_ONE_HAND_THRESHOLD = 5.0    # seconds
HANDS_OFF_BOTH_HANDS_THRESHOLD = 2.0   # seconds
WHEEL_DETECTION_INTERVAL = 1   # seconds
VIDEOMAE_WINDOW_SIZE = 16      # number of frames for action classification
ACTION_OVERLAP = 30             # start new action classification every 30 frames
DEBOUNCE_OFF_FRAMES = 15       # frames (0.5s at 30fps) to confirm hand is off
DEBOUNCE_ON_FRAMES = 3         # frames (0.1s at 30fps) to confirm hand is back on
ACTION_CONFIDENCE_THRESHOLD = 0.5  # confidence threshold for action alerts
EMA_ALPHA = 1.0                # Deprecated, no longer used

# ── Warning severity per action class ──────────────────────────────────────
# Maps model class label strings → warning_type passed to the LLM.
# 'safe_driving' is intentionally omitted — no alert is fired for it.
ACTION_WARNING_MAP: dict[str, str] = {
    # ── High risk — phone / texting ───────────────────────────────────────
    "texting_right"  : "heavy",
    "texting_left"   : "heavy",
    "phonecall_right": "heavy",
    "phonecall_left" : "heavy",
    # ── Medium risk — hands / eyes away from wheel ───────────────────────
    "drinking"       : "light-mid",
    "reach_side"     : "light-mid",
    "hair_and_makeup": "light-mid",
    # ── Low risk — brief / momentary distraction ────────────────────────
    "radio"          : "light",
}

def get_action_warning_type(predicted_class: str) -> str:
    """Return the warning severity for a given model class label.

    Returns None for 'safe_driving' so the caller can skip the alert entirely.
    Falls back to 'moderate' for any class not listed in ACTION_WARNING_MAP.
    """
    if predicted_class == "safe_driving" or predicted_class== "change_gear":
        return None  # no alert for safe behaviour
    return ACTION_WARNING_MAP.get(predicted_class, "moderate")


class ActionRecognitionWorker:
    """Background thread worker for action recognition inference"""
    
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)  # buffer up to 1 frame set
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.running = False
        self.recognizer = None
        self.worker_thread = None
        self.result_id = 0
        
    def start(self):
        """Initialize model and start worker thread"""
        try:
            self.recognizer = ActionRecognizer()
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("[OK] Action recognition worker started")
        except Exception as e:
            print(f"[ERROR] Failed to start action recognizer: {e}")
            self.running = False
    
    def _worker_loop(self):
        """Continuous loop consuming frame buffers"""
        while self.running:
            try:
                item = self.queue.get(timeout=1)
                if item is None:  # shutdown signal
                    break
                frames_buffer = item
                    
                result = self.recognizer.predict(frames_buffer, top_k=3)
                with self.result_lock:
                    self.result_id += 1
                    if result is not None:
                        result["result_id"] = self.result_id
                    self.latest_result = result
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Action recognition error: {e}")
    
    def queue_frames(self, frames: list):
        """Non-blocking push of frame buffer to queue"""
        try:
            self.queue.put_nowait(frames)
        except queue.Full:
            pass  # silently skip if queue full
    
    def get_result(self):
        """Retrieve latest action result (non-blocking)"""
        with self.result_lock:
            return self.latest_result
    
    def stop(self):
        """Gracefully shutdown"""
        self.running = False
        try:
            self.queue.put(None, block=False)
        except queue.Full:
            pass


def classify_action(frames_buffer, action_worker=None):
    """
    Wrapper for action classification.
    If action_worker is provided, queues frames and returns previous result.
    Otherwise returns None (backward compatible).
    """
    if action_worker is None:
        return None
    
    action_worker.queue_frames(frames_buffer.copy())
    return action_worker.get_result()


def build_audio_track(alert_log, total_duration_seconds):
    """
    Mixes all in-memory audio clips into a single raw PCM stream using FFmpeg.
    Each clip is delayed to its alert timestamp, then all are amixed together.
    The result is trimmed to total_duration_seconds so audio never outlasts the video.

    alert_log : list of (video_timestamp_seconds: float, audio_bytes: bytes)
                audio_bytes must be a valid encoded audio file (e.g. MP3 from Groq TTS)
    Returns   : raw PCM bytes (s16le, 44100 Hz, stereo) or None on failure
    """
    if not alert_log:
        return None

    import tempfile
    
    # ── DEBUG: print all alert timestamps ──
    print(f"\n[DEBUG] {len(alert_log)} alert(s) queued for mixing:")
    for i, (ts, audio_bytes) in enumerate(alert_log):
        print(f"   Alert {i+1}: timestamp={ts:.3f}s, size={len(audio_bytes):,} bytes")

    temp_files = []
    try:
        # 1. Write all alert audio bytes to temporary files
        for i, (ts, audio_bytes) in enumerate(alert_log):
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            temp_files.append(tmp.name)

        # 2. Build FFmpeg command
        cmd = [get_ffmpeg_cmd(), "-y"]
        for path in temp_files:
            cmd += ["-i", path]

        filter_parts = []
        for i, (ts, _) in enumerate(alert_log):
            delay_ms = int(ts * 1000)
            filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")

        mix_inputs = "".join(f"[a{i}]" for i in range(len(alert_log)))
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(alert_log)}:normalize=0:dropout_transition=0[aout]"
        )

        cmd += [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[aout]",
            "-t", str(total_duration_seconds),  # hard trim to video length
            "-ar", "44100",
            "-ac", "2",
            "-f", "s16le",                      # raw PCM — no container overhead
            "pipe:1"
        ]

        # 3. Run FFmpeg
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        pcm_bytes, stderr = proc.communicate()

        if proc.returncode != 0:
            print(f"[ERROR] Audio mix failed:\n{stderr.decode(errors='replace')}")
            return None

        print(f"[OK] Mixed audio track: {len(pcm_bytes):,} PCM bytes")
        return pcm_bytes

    except Exception as e:
        print(f"[ERROR] build_audio_track exception: {e}")
        return None
    finally:
        # 4. Clean up temporary files
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as clean_err:
                print(f"[WARN] Failed to clean up temp file {path}: {clean_err}")


def _convert_to_h264(input_path, output_path):
    """Re-encodes a video file to browser-compatible H.264 using FFmpeg."""
    cmd = [
        get_ffmpeg_cmd(), "-y",
        "-i", input_path,
        "-vf", "yadif",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"[OK] Re-encoded {input_path} to H.264: {output_path}")
    except Exception as e:
        print(f"[WARN] H.264 conversion failed: {e}. Falling back to rename.")
        if os.path.exists(input_path):
            try:
                os.rename(input_path, output_path)
            except Exception as re_err:
                print(f"[ERROR] Fallback rename failed: {re_err}")


def mux_audio_into_video(silent_video_path, final_output_path, pcm_bytes, total_duration_seconds):
    """
    Pipes raw PCM bytes into FFmpeg as the audio stream and muxes with the silent video.
    Audio is hard-trimmed to total_duration_seconds. Video is re-encoded to H.264.
    """
    cmd = [
        get_ffmpeg_cmd(), "-y",
        "-i", silent_video_path,    # video from file
        "-f", "s16le",              # raw PCM from stdin
        "-ar", "44100",
        "-ac", "2",
        "-i", "pipe:0",
        "-map", "0:v",
        "-map", "1:a",
        "-vf", "yadif",
        "-c:v", "libx264",          # re-encode to H.264 for browser compatibility
        "-pix_fmt", "yuv420p",      # standard pixel format for web playback
        "-c:a", "aac",
        "-t", str(total_duration_seconds),
        "-shortest",
        final_output_path
    ]

    try:
        subprocess.run(cmd, input=pcm_bytes, capture_output=True, check=True)
        print(f"[OK] Final video with audio saved to: {final_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] FFmpeg mux failed:\n{e.stderr.decode(errors='replace')}")
        _convert_to_h264(silent_video_path, final_output_path)
        print(f"[WARN] Falling back to silent video: {final_output_path}")


def run_pipeline(video_path=None):
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter — silent for now; audio is muxed in at the end
    out = None
    silent_output_path = None
    if video_path:
        silent_output_path = video_path.rsplit(".", 1)[0] + "_annotated_silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(silent_output_path, fourcc, fps, (frame_w, frame_h))
        print(f"Saving annotated video (silent) to: {silent_output_path}")

    # In-memory alert log: list of (video_timestamp_seconds, audio_bytes)
    alert_log = []
    alert_log_lock = threading.Lock()

    # Action recognition worker and EMA smoothing state
    action_worker = ActionRecognitionWorker()
    action_worker.start()
    smoothed_probs = None
    ema_alpha = 0.4

    # State tracking
    hands_off_since = None
    last_roboflow_time = 0
    cached_steering_box = None
    frames_buffer = []
    frame_count = 0
    last_action_frame = -ACTION_OVERLAP
    last_detection_result = None
    last_status_text = []

    # Action recognition state for prioritizing specific alerts over generic hands-off warnings
    last_predicted_action = None
    last_predicted_action_time = 0.0

    # Debounce state (independent counters for left/right hands)
    confirmed_hand_state = {"left": True, "right": True}
    hand_off_frames = {"left": 0, "right": 0}
    hand_on_frames = {"left": 0, "right": 0}

    # Alert threading state
    alert_lock = threading.Lock()
    alert_active = False
    last_alert_finished_video_time = 0.0
    last_alert_distraction_type = None

    def fire_alert(distraction_output, warning_type: str, video_timestamp):
        nonlocal alert_active, last_alert_finished_video_time, last_alert_distraction_type
        distraction_type = distraction_output.get("distraction_type", "distraction")
        try:
            audio_bytes, warning_text = generate_safety_alert_all_groq(distraction_output, warning_type=warning_type)

            if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                with alert_log_lock:
                    alert_log.append((video_timestamp, audio_bytes))
                
                dur = get_wav_duration(audio_bytes)
                last_alert_finished_video_time = video_timestamp + dur
                last_alert_distraction_type = distraction_type
                
                print(f"[INFO] Alert audio captured ({len(audio_bytes):,} bytes, duration={dur:.2f}s) @ {video_timestamp:.2f}s  [warning_type={warning_type!r}]")
                if warning_text:
                    print(f"Spoken text: \"{warning_text}\"")
            else:
                print("[WARN] generate_safety_alert_all_groq returned no bytes")

        except Exception as e:
            print(f"[ERROR] Alert failed: {e}")
        finally:
            with alert_lock:
                alert_active = False

    def trigger_alert(distraction_output, warning_type: str = "moderate"):
        """Fire an audio alert in a background thread.

        Args:
            distraction_output: dict describing what the driver is doing
                                (keys: 'distracted', 'distraction_type').
            warning_type: severity of the alert sent to the LLM/TTS pipeline
                          (e.g. 'light', 'light-mid', 'heavy').
        """
        nonlocal alert_active, last_alert_finished_video_time, last_alert_distraction_type

        distraction_type = distraction_output.get("distraction_type", "distraction")
        video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        with alert_lock:
            if alert_active:
                print(f"   [BLOCKED] Alert already active")
                return False
            
            # Cooldown check
            if last_alert_distraction_type is not None:
                elapsed = video_timestamp - last_alert_finished_video_time
                cooldown = SAME_ACTION_COOLDOWN if distraction_type == last_alert_distraction_type else DIFF_ACTION_COOLDOWN
                if elapsed < cooldown:
                    print(f"   [COOLDOWN BLOCKED] distraction={distraction_type!r}, elapsed={elapsed:.1f}s < cooldown={cooldown}s (last={last_alert_distraction_type!r})")
                    return False
            
            alert_active = True

        print(f"[ALERT] Fired at video_time={video_timestamp:.3f}s  [warning_type={warning_type!r}]")

        threading.Thread(
            target=fire_alert,
            args=(distraction_output, warning_type, video_timestamp),
            daemon=True
        ).start()
        return True
    print("Pipeline running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame_count += 1

        # ── 1. STEERING WHEEL + HAND DETECTION ──
        run_roboflow = False
        if current_time - last_roboflow_time >= 30.0 or cached_steering_box is None:
            run_roboflow = True
            last_roboflow_time = current_time

        # Run MediaPipe on every frame, passing the cached box if Roboflow is skipped
        result = detect_steering_and_hands(frame, steering_box=None if run_roboflow else cached_steering_box)

        if run_roboflow:
            if result["steering_box"] is not None:
                cached_steering_box = result["steering_box"]
            elif last_detection_result is not None and last_detection_result.get("steering_box") is not None:
                result["steering_box"] = last_detection_result["steering_box"]
                cached_steering_box = last_detection_result["steering_box"]
                print("   [WARN] Steering wheel not detected — using last known box as fallback.")
        else:
            result["steering_box"] = cached_steering_box

        last_detection_result = result              

        # ── Debounce logic (independent Left and Right hand checks) ──
        for side in ["left", "right"]:
            is_detected_on = result[f"{side}_hand_on"]
            if is_detected_on:
                hand_on_frames[side] += 1
                hand_off_frames[side] = 0
                if hand_on_frames[side] >= DEBOUNCE_ON_FRAMES:
                    confirmed_hand_state[side] = True
            else:
                hand_off_frames[side] += 1
                hand_on_frames[side] = 0
                if hand_off_frames[side] >= DEBOUNCE_OFF_FRAMES:
                    confirmed_hand_state[side] = False

        # ── Use confirmed_hand_state for alert logic ──
        hands_on_wheel = confirmed_hand_state["left"] and confirmed_hand_state["right"]

        # Check if there is an active warning-worthy action recently detected (in the last 2.0 seconds)
        recent_action_active = (
            last_predicted_action is not None 
            and (current_time - last_predicted_action_time) <= 2.0 
            and get_action_warning_type(last_predicted_action) is not None
        )

        if hands_on_wheel:
            hands_off_since = None
        else:
            if hands_off_since is None:
                hands_off_since = current_time

            hands_off_duration = current_time - hands_off_since
            if frame_count % 15 == 0:
                print(f"Hands off wheel for {hands_off_duration:.1f}s (action_active={recent_action_active})")

            # Only trigger hands-off alerts if no warning-worthy action is active
            if not recent_action_active:
                left_off = not confirmed_hand_state["left"]
                right_off = not confirmed_hand_state["right"]

                # One hand off wheel warning
                if left_off != right_off and hands_off_duration >= HANDS_OFF_ONE_HAND_THRESHOLD:
                    alert_was_fired = trigger_alert(
                        {"distracted": "yes", "distraction_type": "one hand off wheel"},
                        warning_type="light"
                    )
                    hands_off_since = current_time
                    if alert_was_fired:
                        print("[ALERT] Fired one hand off wheel - Blocking triggers until voice finishes.")

                # Both hands off wheel warning
                elif left_off and right_off and hands_off_duration >= HANDS_OFF_BOTH_HANDS_THRESHOLD:
                    alert_was_fired = trigger_alert(
                        {"distracted": "yes", "distraction_type": "both hands off wheel"},
                        warning_type="heavy"
                    )
                    hands_off_since = current_time
                    if alert_was_fired:
                        print("[ALERT] Fired both hands off wheel - Blocking triggers until voice finishes.")

        # ── 2. ACTION CLASSIFICATION ──
        frames_buffer.append(frame)
        if len(frames_buffer) > VIDEOMAE_WINDOW_SIZE:
            frames_buffer.pop(0)

        if (len(frames_buffer) == VIDEOMAE_WINDOW_SIZE and
                frame_count - last_action_frame >= ACTION_OVERLAP):

            last_action_frame = frame_count
            action_result = classify_action(frames_buffer, action_worker)

            if action_result:
                # Apply EMA smoothing to the raw action probabilities
                raw_probs = action_result.get("probs")
                if raw_probs and action_worker.recognizer:
                    if smoothed_probs is None:
                        smoothed_probs = list(raw_probs)
                    else:
                        smoothed_probs = [
                            ema_alpha * curr + (1.0 - ema_alpha) * smooth
                            for curr, smooth in zip(raw_probs, smoothed_probs)
                        ]

                    # Determine the smoothed action and confidence (pure Python argmax)
                    max_idx = smoothed_probs.index(max(smoothed_probs))
                    id2label = action_worker.recognizer.id2label
                    smoothed_action = id2label[max_idx]
                    smoothed_confidence = smoothed_probs[max_idx]
                else:
                    smoothed_action = action_result.get("predicted_class", "unknown")
                    smoothed_confidence = action_result.get("confidence", 0.0)

                # Track predicted action and its timestamp
                if smoothed_confidence >= ACTION_CONFIDENCE_THRESHOLD:
                    last_predicted_action = smoothed_action
                    last_predicted_action_time = current_time

                warning_type    = get_action_warning_type(smoothed_action)

                if warning_type is None:  # safe_driving — no alert
                    print(f"[OK] Safe driving detected (conf={smoothed_confidence:.2f}) — no alert.")
                else:
                    print(
                        f"[WARN] Action detected: '{smoothed_action}' "
                        f"(conf={smoothed_confidence:.2f}, warning_type={warning_type!r})"
                    )
                    trigger_alert(
                        {"distracted": "yes", "distraction_type": smoothed_action},
                        warning_type=warning_type
                    )
                    last_status_text.append((f"⚠ {smoothed_action.upper()}", (0, 165, 255)))

        # ── 3. COMPOSE OUTPUT FRAME ──
        # Re-draw landmarks on the current fresh frame every iteration to avoid stale overlays
        if last_detection_result is not None:
            mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame = draw_landmarks_on_image(mp_frame, last_detection_result["hand_result"])
            output_frame = draw_pose_markers(output_frame, last_detection_result["pose_result"], frame_w, frame_h)
            if last_detection_result["steering_box"]:
                x1, y1, x2, y2 = last_detection_result["steering_box"]
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 8)
                cv2.putText(output_frame, "Steering Wheel", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            output_frame = frame.copy()

        state_y = 100
        for side, key in [("LEFT", "left"), ("RIGHT", "right")]:
            is_on = confirmed_hand_state[key]
            text = f"{side} HAND: {'ON' if is_on else 'OFF'}"
            color = (0, 255, 0) if is_on else (0, 0, 255)
            cv2.putText(output_frame, text, (50, state_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
            state_y += 100

        if pending_hand_state is not None:
            pending_text = (
                f"pending ({pending_count}/{DEBOUNCE_THRESHOLD}): "
                f"L={'ON' if pending_hand_state['left'] else 'OFF'} "
                f"R={'ON' if pending_hand_state['right'] else 'OFF'}"
            )
            cv2.putText(output_frame, pending_text, (50, state_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        for i, (text, color) in enumerate(last_status_text):
            cv2.putText(output_frame, text, (50, frame_h - 60 - i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

        if out:
            out.write(output_frame)

        cv2.imshow("Driver Monitor", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_duration_seconds = frame_count / fps

    cap.release()
    if out:
        out.release()
        print(f"[OK] Silent annotated video saved to: {silent_output_path}")

    cv2.destroyAllWindows()

    # Stop action recognition worker
    action_worker.stop()

    # ── 4. WAIT FOR IN-FLIGHT ALERT THREADS (up to 15 s) ──
    if video_path:
        print("[INFO] Waiting for alert audio threads to finish...")
        deadline = time.time() + 15
        while time.time() < deadline:
            with alert_lock:
                still_active = alert_active
            if not still_active:
                break
            time.sleep(0.2)

    # ── 5. MIX ALL AUDIO IN MEMORY, THEN MUX INTO FINAL VIDEO ──
    if video_path and silent_output_path:
        final_output_path = video_path.rsplit(".", 1)[0] + "_final.mp4"

        with alert_log_lock:
            captured_alerts = list(alert_log)

        if captured_alerts:
            print(f"[INFO] Mixing {len(captured_alerts)} alert(s) entirely in memory...")
            pcm_bytes = build_audio_track(captured_alerts, total_duration_seconds)

            if pcm_bytes:
                mux_audio_into_video(
                    silent_output_path, final_output_path,
                    pcm_bytes, total_duration_seconds
                )
            else:
                print("[WARN] Audio mixing produced no output — encoding silent video to H.264.")
                _convert_to_h264(silent_output_path, final_output_path)
        else:
            print("[INFO] No alerts fired — encoding video to H.264 without audio.")
            _convert_to_h264(silent_output_path, final_output_path)

        # Clean up intermediate silent file
        if os.path.exists(final_output_path) and os.path.exists(silent_output_path):
            os.remove(silent_output_path)


if __name__ == "__main__":
    run_pipeline("test_data/full_videos/gB_10_s2_2019-03-11T15;15;21+01;00_rgb_body.mp4")