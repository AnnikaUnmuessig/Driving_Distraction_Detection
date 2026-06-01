"""
Driver Monitor FastAPI backend.
Provides endpoints for video upload processing, WebSocket video streaming,
job status retrieval, and annotated video downloading.
"""

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from Steering_wheel_detector import detect_steering_and_hands, draw_landmarks_on_image, draw_pose_markers
from Feedback import generate_safety_alert_all_groq

from pipeline import (
    build_audio_track,
    classify_action,
    mux_audio_into_video,
    _convert_to_h264,
    ActionRecognitionWorker,
    ACTION_OVERLAP,
    DEBOUNCE_THRESHOLD,
    HANDS_OFF_THRESHOLD,
    VIDEOMAE_WINDOW_SIZE,
    WHEEL_DETECTION_INTERVAL,
    ACTION_CONFIDENCE_THRESHOLD,
    get_action_warning_type,
)

def get_wav_duration(audio_bytes: bytes) -> float:
    """Returns the duration of WAV audio bytes in seconds, falling back to 2.0s on failure."""
    import wave
    import io
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except Exception:
        return 2.0


UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Driver Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _make_job() -> dict:
    """Creates a new job dictionary structure."""
    return {
        "status": "queued",
        "progress": 0,
        "alert_count": 0,
        "output_path": None,
        "detection_state": None,
        "error": None,
        "frame_queue": queue.Queue(maxsize=120),
        "event_queue": queue.Queue(maxsize=200),
    }


def _update_job(job_id: str, **kwargs):
    """Updates fields of an existing job."""
    with jobs_lock:
        jobs[job_id].update(kwargs)


class DetectionState:
    """Tracks driver state, debouncing, and handles warning alert cooldowns."""
    
    def __init__(self, fps: float = 30.0, play_audio: bool = True):
        """Initializes thresholds, state variables, and spawns the ActionRecognitionWorker."""
        self.fps = fps
        self.play_audio = play_audio
        self.hands_off_since: Optional[float] = None
        self.last_roboflow_time: float = 0
        self.cached_steering_box: Optional[tuple] = None
        self.frames_buffer: list = []
        self.frame_count: int = 0
        self.last_action_time_sec: float = 0.0
        self.action_interval_sec: float = 1.0
        self.last_processed_result_id: int = -1
        self.latest_action_result_time_sec: float = 0.0
        self.last_alert_time: float = 0.0
        self.last_detection_result: Optional[dict] = None
        self.confirmed_hand_state: dict = {"left": True, "right": True}
        self.active_alert_threads: int = 0
        self.alert_lock = threading.Lock()
        self.alert_log: list = []
        self.alert_log_lock = threading.Lock()
        self.latest_action_result: Optional[dict] = None
        self.async_events = queue.Queue()
        self.audio_playing = False
        self.audio_playing_lock = threading.Lock()

        self.same_action_cooldown: float = 5.0
        self.diff_action_cooldown: float = 2.0
        self.last_alert_finished_logical_time: float = 0.0
        self.last_alert_finished_real_time: float = 0.0
        self.last_alert_distraction_type: Optional[str] = None

        self.debounce_off_frames: int = 15
        self.debounce_on_frames: int = 3
        self.hands_off_one_hand_threshold: float = 5.0
        self.hands_off_both_hands_threshold: float = 2.0

        self.hand_off_frames: dict = {"left": 0, "right": 0}
        self.hand_on_frames: dict = {"left": 0, "right": 0}

        self.smoothed_probs = None
        self.ema_alpha = 0.4

        self.action_persist_sec: float = 4.0
        self.action_candidate: Optional[str] = None
        self.action_candidate_since_sec: Optional[float] = None
        self.action_candidate_last_seen_sec: Optional[float] = None

        self.mediapipe_enabled: bool = True
        self.videomae_enabled: bool = True
        self.paused: bool = False
        self.voice_alerts_enabled: bool = True

        self.action_worker = ActionRecognitionWorker()
        self.action_worker.start()

    def _action_candidate_stale(self, now_sec: float) -> bool:
        """Checks if the tracked action candidate has not been seen recently."""
        last = self.action_candidate_last_seen_sec
        if last is None:
            return True
        return (now_sec - last) > max(1.5, self.action_interval_sec * 2.5)

    def _update_action_candidate_from_result(self, action: Optional[str], confidence: float, now_sec: float) -> None:
        """Updates the tracked action candidate or resets it if the action changes or is safe."""
        if not action or confidence <= ACTION_CONFIDENCE_THRESHOLD:
            self.action_candidate = None
            self.action_candidate_since_sec = None
            self.action_candidate_last_seen_sec = None
            return

        if get_action_warning_type(action) is None:
            self.action_candidate = None
            self.action_candidate_since_sec = None
            self.action_candidate_last_seen_sec = None
            return

        if self.action_candidate != action:
            self.action_candidate = action
            self.action_candidate_since_sec = now_sec
            self.action_candidate_last_seen_sec = now_sec
            return

        self.action_candidate_last_seen_sec = now_sec

    def _get_persisted_action(self, now_sec: float) -> Optional[str]:
        """Returns the action candidate if it has persisted continuously for the required duration."""
        if not self.action_candidate or self.action_candidate_since_sec is None:
            return None
        if self._action_candidate_stale(now_sec):
            self.action_candidate = None
            self.action_candidate_since_sec = None
            self.action_candidate_last_seen_sec = None
            return None
        if (now_sec - self.action_candidate_since_sec) >= self.action_persist_sec:
            return self.action_candidate
        return None

    def process_frame(self, frame: np.ndarray, current_time: float, cap_msec: float):
        """Processes a frame, running hands/steering detection, action recognition, and firing alerts."""
        h, w = frame.shape[:2]
        events = []

        while not self.async_events.empty():
            try:
                events.append(self.async_events.get_nowait())
            except queue.Empty:
                break

        time_sec = cap_msec / 1000.0 if not self.play_audio else current_time
        video_time_sec = cap_msec / 1000.0
        logical_time = self.frame_count / self.fps
        real_time = current_time

        action_alert_fired = False
        hand_alert_pending = None

        if self.mediapipe_enabled:
            run_roboflow = False
            if time_sec - self.last_roboflow_time >= 30.0 or self.cached_steering_box is None:
                run_roboflow = True
                self.last_roboflow_time = time_sec

            result = detect_steering_and_hands(frame, steering_box=None if run_roboflow else self.cached_steering_box)

            if run_roboflow:
                if result["steering_box"] is not None:
                    self.cached_steering_box = result["steering_box"]
                elif self.last_detection_result is not None and self.last_detection_result.get("steering_box") is not None:
                    result["steering_box"] = self.last_detection_result["steering_box"]
                    self.cached_steering_box = self.last_detection_result["steering_box"]
            else:
                result["steering_box"] = self.cached_steering_box

            self.last_detection_result = result
            old_state = self.confirmed_hand_state.copy()

            for side in ["left", "right"]:
                is_detected_on = result[f"{side}_hand_on"]
                if is_detected_on:
                    self.hand_on_frames[side] += 1
                    self.hand_off_frames[side] = 0
                    if self.hand_on_frames[side] >= self.debounce_on_frames:
                        self.confirmed_hand_state[side] = True
                else:
                    self.hand_off_frames[side] += 1
                    self.hand_on_frames[side] = 0
                    if self.hand_off_frames[side] >= self.debounce_off_frames:
                        self.confirmed_hand_state[side] = False

            if old_state != self.confirmed_hand_state:
                events.append({
                    "type": "hand_state",
                    "confirmed": self.confirmed_hand_state,
                    "pending_count": 0
                })

            hands_on = self.confirmed_hand_state["left"] and self.confirmed_hand_state["right"]
            if hands_on:
                self.hands_off_since = None
            else:
                if self.hands_off_since is None:
                    self.hands_off_since = time_sec
                off_duration = time_sec - self.hands_off_since
                left_off  = not self.confirmed_hand_state["left"]
                right_off = not self.confirmed_hand_state["right"]

                if left_off != right_off and off_duration >= self.hands_off_one_hand_threshold:
                    hand_alert_pending = (
                        {"distracted": "yes", "distraction_type": "one hand off wheel"},
                        "light"
                    )
                elif left_off and right_off and off_duration >= self.hands_off_both_hands_threshold:
                    hand_alert_pending = (
                        {"distracted": "yes", "distraction_type": "both hands off wheel"},
                        "heavy"
                    )

        if self.videomae_enabled:
            self.frames_buffer.append(frame)
            max_buffer_size = max(16, int(round(self.fps * self.action_interval_sec)))
            while len(self.frames_buffer) > max_buffer_size:
                self.frames_buffer.pop(0)

            if (
                len(self.frames_buffer) >= 16
                and time_sec - self.last_action_time_sec >= self.action_interval_sec
            ):
                self.last_action_time_sec = time_sec
                n = len(self.frames_buffer)
                indices = np.linspace(0, n - 1, 16, dtype=int)
                sampled_frames = [self.frames_buffer[i] for i in indices]
                self.action_worker.queue_frames(sampled_frames)

            action_result = self.action_worker.get_result()
            if action_result:
                result_id = action_result.get("result_id", 0)
                if result_id > self.last_processed_result_id:
                    self.last_processed_result_id = result_id
                    self.latest_action_result_time_sec = time_sec

                    raw_probs = action_result.get("probs")
                    if raw_probs and self.action_worker.recognizer:
                        if self.smoothed_probs is None:
                            self.smoothed_probs = list(raw_probs)
                        else:
                            self.smoothed_probs = [
                                self.ema_alpha * curr + (1.0 - self.ema_alpha) * smooth
                                for curr, smooth in zip(raw_probs, self.smoothed_probs)
                            ]

                        max_idx = int(np.argmax(self.smoothed_probs))
                        id2label = self.action_worker.recognizer.id2label
                        smoothed_action = id2label[max_idx]
                        smoothed_confidence = self.smoothed_probs[max_idx]

                        sorted_indices = sorted(range(len(self.smoothed_probs)), key=lambda i: self.smoothed_probs[i], reverse=True)
                        self.latest_action_result = {
                            "predicted_class": smoothed_action,
                            "confidence": round(smoothed_confidence, 4),
                            "top_k": [(id2label[i], round(self.smoothed_probs[i], 4)) for i in sorted_indices[:3]],
                            "result_id": result_id
                        }

                        self._update_action_candidate_from_result(smoothed_action, smoothed_confidence, time_sec)
                    else:
                        action = action_result.get("predicted_class")
                        confidence = float(action_result.get("confidence", 0) or 0)
                        self.latest_action_result = action_result
                        self._update_action_candidate_from_result(action, confidence, time_sec)

                    persisted_action = self._get_persisted_action(time_sec)
                    if persisted_action:
                        warning_type = get_action_warning_type(persisted_action)
                        if warning_type is not None:
                            fired = self._trigger_alert(
                                {"distracted": "yes", "distraction_type": persisted_action},
                                warning_type=warning_type,
                                logical_time=logical_time,
                                real_time=real_time,
                                trigger_source="action",
                                video_time_sec=video_time_sec,
                            )
                            if fired:
                                action_alert_fired = True
                                events.append({
                                    "type": "alert",
                                    "distraction_type": persisted_action,
                                    "severity": warning_type,
                                    "trigger_source": "action",
                                })

        if hand_alert_pending and not action_alert_fired:
            with self.audio_playing_lock:
                audio_busy = self.audio_playing
            if not audio_busy:
                recent_action_active = (
                    self.action_candidate is not None
                    and get_action_warning_type(self.action_candidate) is not None
                )

                if recent_action_active:
                    self.frame_count += 1
                    output_frame = self._annotate(frame, w, h)
                    return {"frame": output_frame, "events": events}

                distraction_output, warning_type = hand_alert_pending
                fired = self._trigger_alert(
                    distraction_output,
                    warning_type=warning_type,
                    logical_time=logical_time,
                    real_time=real_time,
                    trigger_source="hands_off",
                    video_time_sec=video_time_sec,
                )
                if fired:
                    self.hands_off_since = time_sec
                    events.append({
                        "type": "alert",
                        "distraction_type": distraction_output["distraction_type"],
                        "severity": warning_type,
                        "trigger_source": "hands_off",
                    })
            self.frame_count += 1
        output_frame = self._annotate(frame, w, h)
        return {"frame": output_frame, "events": events}

    def _trigger_alert(self, distraction_output, warning_type: str, logical_time: float, real_time: float, trigger_source: str, video_time_sec: float):
        """Checks cooldowns and active playback to safely initiate alert thread."""
        distraction_type = distraction_output.get("distraction_type", "distraction")
        with self.alert_lock:
            if self.last_alert_distraction_type is not None:
                cooldown = self.same_action_cooldown if distraction_type == self.last_alert_distraction_type else self.diff_action_cooldown
                
                logical_elapsed = logical_time - self.last_alert_finished_logical_time
                if logical_elapsed < cooldown:
                    print(f"   [COOLDOWN BLOCKED LOGICAL] distraction={distraction_type!r}, logical_elapsed={logical_elapsed:.2f}s < cooldown={cooldown}s")
                    return False
                
                if self.play_audio:
                    real_elapsed = real_time - self.last_alert_finished_real_time
                    if real_elapsed < cooldown:
                        print(f"   [COOLDOWN BLOCKED REAL] distraction={distraction_type!r}, real_elapsed={real_elapsed:.2f}s < cooldown={cooldown}s")
                        return False
            
            if self.play_audio:
                with self.audio_playing_lock:
                    if self.audio_playing:
                        print("[BLOCKED] Audio still playing on speaker")
                        return False
                    self.audio_playing = True

            self.last_alert_finished_logical_time = logical_time + 3.0
            self.last_alert_finished_real_time = real_time + 3.0
            self.last_alert_distraction_type = distraction_type
            self.active_alert_threads += 1

        print(f"[DEBUG] Spawning alert thread at logical_time={logical_time:.2f}s")
        threading.Thread(
            target=self._fire_alert,
            args=(distraction_output, warning_type, logical_time, real_time, self.play_audio, trigger_source, video_time_sec),
            daemon=False
        ).start()
        return True

    def _fire_alert(self, distraction_output, warning_type: str, logical_time: float, real_time: float, play_audio: bool, trigger_source: str, video_time_sec: float):
        """Requests alert audio generation, updates cooldowns, and plays audio."""
        distraction_type = distraction_output.get("distraction_type", "distraction")
        try:
            gen_audio = self.voice_alerts_enabled
            should_play = play_audio and self.voice_alerts_enabled

            audio_bytes, warning_text = generate_safety_alert_all_groq(
                distraction_output, warning_type=warning_type, play_audio=should_play, generate_audio=gen_audio
            )
            dur = 2.0
            if gen_audio and isinstance(audio_bytes, bytes) and audio_bytes:
                with self.alert_log_lock:
                    self.alert_log.append((video_time_sec, audio_bytes))
                dur = get_wav_duration(audio_bytes)
            
            self.last_alert_finished_logical_time = logical_time + dur
            self.last_alert_finished_real_time = time.time()
            self.last_alert_distraction_type = distraction_type

            if warning_text:
                self.async_events.put({
                    "type": "alert_text",
                    "text": warning_text,
                    "distraction_type": distraction_output.get("distraction_type"),
                    "severity": warning_type,
                    "trigger_source": trigger_source,
                })
        except Exception as e:
            print(f"Alert error: {e}")
        finally:
            if play_audio:
                with self.audio_playing_lock:
                    self.audio_playing = False
            with self.alert_lock:
                self.active_alert_threads -= 1

    def _annotate(self, frame, w, h):
        """Draws bounding boxes, hand/pose landmarks, and prediction overlays on the frame."""
        out = frame.copy()

        if self.mediapipe_enabled and self.last_detection_result is not None:
            r = self.last_detection_result
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            out = draw_landmarks_on_image(rgb, r["hand_result"])
            out = draw_pose_markers(out, r["pose_result"], w, h)

            if r["steering_box"]:
                x1, y1, x2, y2 = r["steering_box"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(out, "Steering Wheel", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                label_x = x1 + 8
                label_y = y2 - 10
                for side, key in [("Left Hand", "left"), ("Right Hand", "right")]:
                    is_on = self.confirmed_hand_state[key]
                    color = (0, 255, 0) if is_on else (0, 0, 255)
                    text = f"{side}: {'ON' if is_on else 'OFF'}"
                    cv2.putText(out, text, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    label_y -= 24
            else:
                label_y = 30
                for side, key in [("Left Hand", "left"), ("Right Hand", "right")]:
                    is_on = self.confirmed_hand_state[key]
                    color = (0, 255, 0) if is_on else (0, 0, 255)
                    text = f"{side}: {'ON' if is_on else 'OFF'}"
                    cv2.putText(out, text, (10, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    label_y += 24

        if self.videomae_enabled and self.latest_action_result:
            ar = self.latest_action_result
            label = ar["predicted_class"].replace("_", " ").title()
            confidence = ar["confidence"]
            top3 = ar.get("top_k", [])

            overlay = out.copy()
            box_h = 90
            cv2.rectangle(overlay, (0, h - box_h), (w, h), (20, 20, 20), -1)
            out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

            cv2.putText(out, label, (12, h - box_h + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            conf_text = f"{confidence * 100:.1f}%"
            cv2.putText(out, conf_text, (12, h - box_h + 58),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 230, 100), 1, cv2.LINE_AA)

            bar_x = w - 240
            for rank, (cls, score) in enumerate(top3[:3]):
                bar_y = h - box_h + 18 + rank * 24
                bar_len = int(score * 180)
                bar_col = (100, 230, 100) if rank == 0 else (180, 180, 180)
                cv2.rectangle(out, (bar_x, bar_y - 10), (bar_x + bar_len, bar_y + 4), bar_col, -1)
                short_cls = cls.replace("_", " ")[:18]
                cv2.putText(out, f"{short_cls} {score*100:.0f}%",
                            (bar_x, bar_y + 3), cv2.FONT_HERSHEY_DUPLEX, 0.38,
                            (255, 255, 255), 1, cv2.LINE_AA)

        return out


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Receives a video file and schedules background pipeline processing."""
    job_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    job = _make_job()
    with jobs_lock:
        jobs[job_id] = job

    threading.Thread(target=_run_upload_pipeline, args=(job_id, str(save_path)), daemon=True).start()
    return {"job_id": job_id}


def _run_upload_pipeline(job_id: str, video_path: str):
    """Processes video in a background thread, writing output to disk and queues."""
    with jobs_lock:
        job = jobs[job_id]
    fq: queue.Queue = job["frame_queue"]
    eq: queue.Queue = job["event_queue"]

    try:
        _update_job(job_id, status="processing")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_job(job_id, status="error", error="Could not open video file")
            fq.put(None)
            eq.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        state = DetectionState(fps=fps, play_audio=True)
        with jobs_lock:
            job["detection_state"] = state

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        silent_path = str(OUTPUT_DIR / f"{job_id}_silent.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out_writer = cv2.VideoWriter(silent_path, fourcc, fps, (w, h))

        frame_count = 0
        while cap.isOpened():
            if state.paused:
                time.sleep(0.1)
                continue
            ret, frame = cap.read()
            if not ret:
                break

            result = state.process_frame(frame, time.time(), cap.get(cv2.CAP_PROP_POS_MSEC))
            annotated = result["frame"]

            out_writer.write(annotated)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            try:
                fq.put_nowait(buf.tobytes())
            except queue.Full:
                pass

            for event in result["events"]:
                try:
                    eq.put_nowait(event)
                except queue.Full:
                    pass

            frame_count += 1
            progress = min(99, int(frame_count / total_frames * 100))
            _update_job(job_id, progress=progress, alert_count=len(state.alert_log))

        cap.release()
        out_writer.release()

        total_duration = frame_count / fps

        deadline = time.time() + 15
        while time.time() < deadline:
            with state.alert_lock:
                still = state.active_alert_threads > 0
            if not still:
                break
            time.sleep(0.2)

        final_path = str(OUTPUT_DIR / f"{job_id}_final.mp4")
        with state.alert_log_lock:
            alerts = list(state.alert_log)

        if alerts:
            pcm = build_audio_track(alerts, total_duration)
            if pcm:
                mux_audio_into_video(silent_path, final_path, pcm, total_duration)
            else:
                _convert_to_h264(silent_path, final_path)
        else:
            _convert_to_h264(silent_path, final_path)

        if os.path.exists(final_path) and os.path.exists(silent_path):
            os.remove(silent_path)

        _update_job(job_id, status="done", progress=100, output_path=final_path, alert_count=len(alerts))

    except Exception as e:
        _update_job(job_id, status="error", error=str(e))
    finally:
        try:
            fq.put_nowait(None)
        except queue.Full:
            pass
        try:
            eq.put_nowait(None)
        except queue.Full:
            pass


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Retrieves basic status information about an upload job."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {k: v for k, v in job.items() if k not in ("frame_queue", "event_queue", "detection_state")} | {"job_id": job_id}


@app.get("/output/{job_id}")
def download_output(job_id: str):
    """Downloads the completed annotated video file for a job."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Output not ready")
    return FileResponse(job["output_path"], media_type="video/mp4", filename=f"annotated_{job_id}.mp4")


@app.websocket("/stream/upload/{job_id}")
async def stream_upload_frames(websocket: WebSocket, job_id: str):
    """Consumes frame and event queues from background processing thread and streams to websocket client."""
    await websocket.accept()

    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        await websocket.send_json({"type": "error", "message": "Job not found"})
        await websocket.close()
        return

    fq: queue.Queue = job["frame_queue"]
    eq: queue.Queue = job["event_queue"]
    loop = asyncio.get_event_loop()

    async def _listen_toggles():
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("type") == "toggle":
                    with jobs_lock:
                        ds = job.get("detection_state")
                    if ds:
                        target = data.get("target")
                        enabled = data.get("enabled", True)
                        if target == "mediapipe":
                            ds.mediapipe_enabled = enabled
                        elif target == "videomae":
                            ds.videomae_enabled = enabled
                        elif target == "audio":
                            ds.voice_alerts_enabled = enabled
                elif data.get("type") == "pause":
                    with jobs_lock:
                        ds = job.get("detection_state")
                    if ds:
                        ds.paused = data.get("paused", False)
                elif data.get("type") == "config":
                    with jobs_lock:
                        ds = job.get("detection_state")
                    if ds:
                        key = data.get("key")
                        val = data.get("value")
                        if key == "action_interval":
                            ds.action_interval_sec = float(val)
        except (WebSocketDisconnect, Exception):
            pass

    listener = asyncio.create_task(_listen_toggles())

    try:
        while True:
            while True:
                try:
                    event = eq.get_nowait()
                    if event is None:
                        break
                    await websocket.send_json(event)
                except queue.Empty:
                    break

            try:
                frame_bytes = await asyncio.wait_for(
                    loop.run_in_executor(None, fq.get),
                    timeout=600.0,
                )
            except asyncio.TimeoutError:
                break

            if frame_bytes is None:
                await websocket.send_json({"type": "done"})
                break

            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()


@app.websocket("/stream/webcam")
async def stream_webcam(websocket: WebSocket, cam_index: int = 0):
    """Runs pipeline in real-time on live camera feed and streams frames and events to websocket client."""
    await websocket.accept()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"Cannot open camera {cam_index}"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    state = DetectionState(fps=fps, play_audio=True)

    async def _listen_toggles():
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("type") == "toggle":
                    target = data.get("target")
                    enabled = data.get("enabled", True)
                    if target == "mediapipe":
                        state.mediapipe_enabled = enabled
                    elif target == "videomae":
                        state.videomae_enabled = enabled
                    elif target == "audio":
                        state.play_audio = enabled
                        state.voice_alerts_enabled = enabled
                elif data.get("type") == "config":
                    key = data.get("key")
                    val = data.get("value")
                    if key == "action_interval":
                        state.action_interval_sec = float(val)
        except (WebSocketDisconnect, Exception):
            pass

    listener = asyncio.create_task(_listen_toggles())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.05)
                continue

            result = state.process_frame(frame, time.time(), cap.get(cv2.CAP_PROP_POS_MSEC))
            _, buf = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 75])
            await websocket.send_bytes(buf.tobytes())

            for event in result["events"]:
                await websocket.send_json(event)

            await asyncio.sleep(frame_delay)

    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()
        cap.release()
