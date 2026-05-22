# This is the final pipeline
import cv2
import time
import threading
import os
import subprocess
import queue
from Steering_wheel_detector import detect_steering_and_hands, draw_landmarks_on_image, draw_pose_markers
from Feedback import generate_safety_alert_all_groq
from action_recognition import ActionRecognizer

# Fixed variables
HANDS_OFF_THRESHOLD = 1        # seconds temporarily low
WHEEL_DETECTION_INTERVAL = 1   # seconds
TIMESFORMER_WINDOW_SIZE = 16   # number of frames for action classification
ACTION_OVERLAP = 8             # start new action classification every 8 frames
DEBOUNCE_THRESHOLD = 3         # number of consecutive detections to confirm state change
ACTION_CONFIDENCE_THRESHOLD = 0.5  # confidence threshold for action alerts


class ActionRecognitionWorker:
    """Background thread worker for action recognition inference"""
    
    def __init__(self):
        self.queue = queue.Queue(maxsize=5)  # buffer up to 5 frame sets
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.running = False
        self.recognizer = None
        self.worker_thread = None
        
    def start(self):
        """Initialize model and start worker thread"""
        try:
            self.recognizer = ActionRecognizer()
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("✅ Action recognition worker started")
        except Exception as e:
            print(f"❌ Failed to start action recognizer: {e}")
            self.running = False
    
    def _worker_loop(self):
        """Continuous loop consuming frame buffers"""
        while self.running:
            try:
                frames_buffer = self.queue.get(timeout=1)
                if frames_buffer is None:  # shutdown signal
                    break
                    
                result = self.recognizer.predict(frames_buffer, top_k=3)
                with self.result_lock:
                    self.latest_result = result
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Action recognition error: {e}")
    
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
    # ── DEBUG: print all alert timestamps ──
    print(f"\n🔍 DEBUG — {len(alert_log)} alert(s) queued for mixing:")
    for i, (ts, audio_bytes) in enumerate(alert_log):
        print(f"   Alert {i+1}: timestamp={ts:.3f}s, size={len(audio_bytes):,} bytes")


    # One anonymous OS pipe per clip — FFmpeg reads the read-end, parent writes the write-end
    pipe_pairs = []
    for _ in alert_log:
        r, w = os.pipe()
        pipe_pairs.append((r, w))

    read_fds = [r for r, _ in pipe_pairs]

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y"]
    for r_fd in read_fds:
        cmd += ["-i", f"pipe:{r_fd}"]

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

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=tuple(read_fds)
        )

        # Write each clip into its pipe's write-end in a background thread.
        # Must be threaded to avoid deadlocking while FFmpeg reads simultaneously.
        def write_clip(w_fd, audio_bytes):
            try:
                with os.fdopen(w_fd, "wb") as f:
                    f.write(audio_bytes)
            except BrokenPipeError:
                pass  # FFmpeg finished early (e.g. clip longer than -t) — that's fine

        writers = []
        for (r_fd, w_fd), (_, audio_bytes) in zip(pipe_pairs, alert_log):
            os.close(r_fd)  # close the read-end in the parent; FFmpeg holds its copy
            t = threading.Thread(target=write_clip, args=(w_fd, audio_bytes), daemon=True)
            t.start()
            writers.append(t)

        pcm_bytes, stderr = proc.communicate()

        for t in writers:
            t.join(timeout=5)

        if proc.returncode != 0:
            print(f"❌ Audio mix failed:\n{stderr.decode(errors='replace')}")
            return None

        print(f"✅ Mixed audio track: {len(pcm_bytes):,} PCM bytes")
        return pcm_bytes

    except Exception as e:
        print(f"❌ build_audio_track exception: {e}")
        for r, w in pipe_pairs:
            for fd in (r, w):
                try:
                    os.close(fd)
                except OSError:
                    pass
        return None


def mux_audio_into_video(silent_video_path, final_output_path, pcm_bytes, total_duration_seconds):
    """
    Pipes raw PCM bytes into FFmpeg as the audio stream and muxes with the silent video.
    Audio is hard-trimmed to total_duration_seconds. No temp files are written.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,    # video from file
        "-f", "s16le",              # raw PCM from stdin
        "-ar", "44100",
        "-ac", "2",
        "-i", "pipe:0",
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",             # no video re-encode
        "-c:a", "aac",
        "-t", str(total_duration_seconds),
        "-shortest",
        final_output_path
    ]

    try:
        subprocess.run(cmd, input=pcm_bytes, capture_output=True, check=True)
        print(f"✅ Final video with audio saved to: {final_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg mux failed:\n{e.stderr.decode(errors='replace')}")
        os.rename(silent_video_path, final_output_path)
        print(f"⚠️  Falling back to silent video: {final_output_path}")


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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(silent_output_path, fourcc, fps, (frame_w, frame_h))
        print(f"Saving annotated video (silent) to: {silent_output_path}")

    # In-memory alert log: list of (video_timestamp_seconds, audio_bytes)
    alert_log = []
    alert_log_lock = threading.Lock()

    # State tracking
    hands_off_since = None
    last_wheel_check_time = 0
    frames_buffer = []
    frame_count = 0
    last_action_frame = -ACTION_OVERLAP
    last_detection_result = None
    last_status_text = []

    # Debounce state
    confirmed_hand_state = {"left": True, "right": True}
    pending_hand_state = None
    pending_count = 0

    # Alert threading state
    alert_lock = threading.Lock()
    alert_active = False

    def fire_alert(distraction_output, video_timestamp):
        nonlocal alert_active
        try:
            audio_bytes = generate_safety_alert_all_groq(distraction_output)

            if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                with alert_log_lock:
                    alert_log.append((video_timestamp, audio_bytes))
                print(f"🔊 Alert audio captured ({len(audio_bytes):,} bytes) @ {video_timestamp:.2f}s")
            else:
                print("⚠️  generate_safety_alert_all_groq returned no bytes")

        except Exception as e:
            print(f"❌ Alert failed: {e}")
        finally:
            with alert_lock:
                alert_active = False

    def trigger_alert(distraction_output):
        nonlocal alert_active

        with alert_lock:
            if alert_active:
                print(f"   ❌ Blocked — alert already active")
                return False
            alert_active = True
            video_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        print(f"🚨 Alert fired at video_time={video_timestamp:.3f}s")

        threading.Thread(
            target=fire_alert,
            args=(distraction_output, video_timestamp),
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
        # ── 1. STEERING WHEEL + HAND DETECTION ──
        if current_time - last_wheel_check_time >= WHEEL_DETECTION_INTERVAL:
            last_wheel_check_time = current_time

            # Current detection call
            result = detect_steering_and_hands(frame)
 
            # If current detection returned no box, but we have a previous result,
            # pull the steering_box from that last_detection_result.
            if result["steering_box"] is None and last_detection_result is not None:
                if last_detection_result.get("steering_box") is not None:
                    result["steering_box"] = last_detection_result["steering_box"]
                    print("   ⚠️  Steering wheel not detected — using last known box as fallback.")
                    # Optionally, you could print a log here to know it's falling back
                    # print("Using cached steering wheel box")

            # Now, update the cache with this 'result' (which now contains the old box if needed)
            last_detection_result = result              

            new_state = {
                "left": result["left_hand_on"],
                "right": result["right_hand_on"]
            }

            # ── Debounce logic ──
            if new_state == confirmed_hand_state:
                pending_hand_state = None
                pending_count = 0
            elif new_state == pending_hand_state:
                pending_count += 1
                if pending_count >= DEBOUNCE_THRESHOLD:
                    print(f"State change confirmed: {confirmed_hand_state} → {new_state}")
                    confirmed_hand_state = new_state
                    pending_hand_state = None
                    pending_count = 0
            else:
                pending_hand_state = new_state
                pending_count = 1

            # ── Use confirmed_hand_state for alert logic ──
            hands_on_wheel = confirmed_hand_state["left"] and confirmed_hand_state["right"]

            if hands_on_wheel:
                hands_off_since = None
            else:
                if hands_off_since is None:
                    hands_off_since = current_time

                hands_off_duration = current_time - hands_off_since
                print(f"Hands off wheel for {hands_off_duration:.1f}s")

                if hands_off_duration >= HANDS_OFF_THRESHOLD:
                    alert_was_fired = trigger_alert({
                        "distracted": "yes",
                        "distraction_type": "hands off wheel",
                        "type of warning": "mid-heavy"
                    })
                    hands_off_since = current_time

                    if alert_was_fired:
                        print("📢 Alert started - Blocking new triggers until voice finishes.")

        # ── 2. ACTION CLASSIFICATION ──
        frames_buffer.append(frame)
        if len(frames_buffer) > TIMESFORMER_WINDOW_SIZE:
            frames_buffer.pop(0)

        if (len(frames_buffer) == TIMESFORMER_WINDOW_SIZE and
                frame_count - last_action_frame >= ACTION_OVERLAP):

            last_action_frame = frame_count
            action = classify_action(frames_buffer)

            if action:
                print(f"⚠️  Distraction detected: {action}")
                trigger_alert({
                    "distracted": "yes",
                    "distraction_type": action,
                    "type of warning": "light-mid"
                })
                last_status_text.append((f"⚠ {action.upper()}", (0, 165, 255)))

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
        print(f"✅ Silent annotated video saved to: {silent_output_path}")

    cv2.destroyAllWindows()

    # ── 4. WAIT FOR IN-FLIGHT ALERT THREADS (up to 15 s) ──
    if video_path:
        print("⏳ Waiting for alert audio threads to finish...")
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
            print(f"🎚️  Mixing {len(captured_alerts)} alert(s) entirely in memory...")
            pcm_bytes = build_audio_track(captured_alerts, total_duration_seconds)

            if pcm_bytes:
                mux_audio_into_video(
                    silent_output_path, final_output_path,
                    pcm_bytes, total_duration_seconds
                )
            else:
                print("⚠️  Audio mixing produced no output — saving silent video.")
                os.rename(silent_output_path, final_output_path)
        else:
            print("ℹ️  No alerts fired — saving video without audio.")
            os.rename(silent_output_path, final_output_path)

        # Clean up intermediate silent file
        if os.path.exists(final_output_path) and os.path.exists(silent_output_path):
            os.remove(silent_output_path)


run_pipeline("test_data/full_videos/gB_10_s2_2019-03-11T15;15;21+01;00_rgb_body.mp4")