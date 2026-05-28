"""
Driver Monitor — FastAPI backend
Exposes:
  POST /upload                      — accepts video, runs pipeline in background thread
  GET  /stream/upload/{job_id}      — WebSocket, streams frames produced by the pipeline thread
  GET  /stream/webcam               — WebSocket, runs pipeline on cv2.VideoCapture(cam_index)
  GET  /jobs/{job_id}               — poll status + progress
  GET  /output/{job_id}             — download the _final.mp4 once ready
"""

from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
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

# ── job store ──────────────────────────────────────────────────────────────────
# Each job gets:
#   status, progress, alert_count, output_path, error
#   frame_queue: Queue[bytes | None]  — JPEG bytes pushed by pipeline thread;
#                                        None signals end-of-stream
#   event_queue: Queue[dict | None]   — alert JSON events; None = done
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _make_job() -> dict:
    return {
        "status": "queued",
        "progress": 0,
        "alert_count": 0,
        "output_path": None,
        "detection_state": None,
        "error": None,
        "frame_queue": queue.Queue(maxsize=120),   # ~4 s buffer at 30 fps
        "event_queue": queue.Queue(maxsize=200),
    }


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


# ── shared detection state ─────────────────────────────────────────────────────
class DetectionState:
    def __init__(self, fps: float = 30.0, play_audio: bool = True):
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
        self.pending_hand_state: Optional[dict] = None
        self.pending_count: int = 0
        self.active_alert_threads: int = 0
        self.alert_lock = threading.Lock()
        self.alert_log: list = []
        self.alert_log_lock = threading.Lock()
        self.latest_action_result: Optional[dict] = None
        self.async_events = queue.Queue()
        self.audio_playing = False
        self.audio_playing_lock = threading.Lock()

        # Action persistence: only warn if the same warning-worthy action persists.
        self.action_persist_sec: float = 4.0
        self.action_candidate: Optional[str] = None
        self.action_candidate_since_sec: Optional[float] = None
        self.action_candidate_last_seen_sec: Optional[float] = None

        # Toggle flags (controllable from frontend)
        self.mediapipe_enabled: bool = True
        self.videomae_enabled: bool = True
        self.paused: bool = False


        # Action recognition worker
        self.action_worker = ActionRecognitionWorker()
        self.action_worker.start()

    def _action_candidate_stale(self, now_sec: float) -> bool:
        last = self.action_candidate_last_seen_sec
        if last is None:
            return True
        # Allow slack so we don't reset between model updates.
        return (now_sec - last) > max(1.5, self.action_interval_sec * 2.5)

    def _update_action_candidate_from_result(self, action: Optional[str], confidence: float, now_sec: float) -> None:
        if not action or confidence <= ACTION_CONFIDENCE_THRESHOLD:
            self.action_candidate = None
            self.action_candidate_since_sec = None
            self.action_candidate_last_seen_sec = None
            return

        # Only persist "warning-worthy" actions; safe_driving and other safe actions don't count.
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
        h, w = frame.shape[:2]
        events = []

        # Drain async events queue
        while not self.async_events.empty():
            try:
                events.append(self.async_events.get_nowait())
            except queue.Empty:
                break

        time_sec = cap_msec / 1000.0 if not self.play_audio else current_time

        action_alert_fired = False  # track if action fired this frame
        hand_alert_pending = None   # store hand alert to fire later

        # -- Mediapipe: steering wheel + hand detection --
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
            new_state = {"left": result["left_hand_on"], "right": result["right_hand_on"]}

            if new_state == self.confirmed_hand_state:
                self.pending_hand_state = None
                self.pending_count = 0
            elif new_state == self.pending_hand_state:
                self.pending_count += 1
                if self.pending_count >= DEBOUNCE_THRESHOLD:
                    old_state = self.confirmed_hand_state.copy()
                    self.confirmed_hand_state = new_state
                    self.pending_hand_state = None
                    self.pending_count = 0
                    if old_state != self.confirmed_hand_state:
                        events.append({
                            "type": "hand_state",
                            "confirmed": self.confirmed_hand_state,
                            "pending_count": self.pending_count
                        })
            else:
                self.pending_hand_state = new_state
                self.pending_count = 1
                events.append({
                    "type": "hand_state",
                    "confirmed": self.confirmed_hand_state,
                    "pending_count": self.pending_count
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

                # Store hand alert as pending — don't fire yet
                if off_duration >= HANDS_OFF_THRESHOLD and left_off != right_off:
                    hand_alert_pending = (
                        {"distracted": "yes", "distraction_type": "one hand off wheel"},
                        "light"
                    )
                elif off_duration >= HANDS_OFF_THRESHOLD and left_off and right_off:
                    hand_alert_pending = (
                        {"distracted": "yes", "distraction_type": "both hands off wheel"},
                        "heavy"
                    )

        # -- VideoMAE: action classification --
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

            result = self.action_worker.get_result()
            if result:
                result_id = result.get("result_id", 0)
                if result_id > self.last_processed_result_id:
                    self.last_processed_result_id = result_id
                    self.latest_action_result = result
                    self.latest_action_result_time_sec = time_sec

                    action = result.get("predicted_class")
                    confidence = float(result.get("confidence", 0) or 0)
                    self._update_action_candidate_from_result(action, confidence, time_sec)

                    persisted_action = self._get_persisted_action(time_sec)
                    if persisted_action:
                        warning_type = get_action_warning_type(persisted_action)
                        if warning_type is not None:
                            fired = self._trigger_alert(
                                {"distracted": "yes", "distraction_type": persisted_action},
                                warning_type=warning_type,
                                timestamp_sec=time_sec,
                                trigger_source="action",
                            )
                            if fired:
                                action_alert_fired = True  # mark action as fired
                                events.append({
                                    "type": "alert",
                                    "distraction_type": persisted_action,
                                    "severity": warning_type,
                                    "trigger_source": "action",
                                })

        # -- Fire hand alert only if action didn't fire this frame AND no action alert is currently playing --
        if hand_alert_pending and not action_alert_fired:
            with self.audio_playing_lock:
                audio_busy = self.audio_playing
            if not audio_busy:
                # If hands are off wheel:
                # - If current action == safe_driving -> warn hands-off
                # - Else -> only warn once a warning-worthy action persists for action_persist_sec seconds
                ar = self.latest_action_result
                recent_action = (
                    ar is not None
                    and (time_sec - self.latest_action_result_time_sec)
                    <= max(0.25, self.action_interval_sec * 1.25)
                )
                current_action = ar.get("predicted_class") if (ar and recent_action) else None

                if current_action and current_action != "safe_driving":
                    persisted_action = self._get_persisted_action(time_sec)
                    if persisted_action:
                        warning_type = get_action_warning_type(persisted_action)
                        if warning_type is not None:
                            fired = self._trigger_alert(
                                {"distracted": "yes", "distraction_type": persisted_action},
                                warning_type=warning_type,
                                timestamp_sec=time_sec,
                                trigger_source="action",
                            )
                            if fired:
                                action_alert_fired = True
                                events.append({
                                    "type": "alert",
                                    "distraction_type": persisted_action,
                                    "severity": warning_type,
                                    "trigger_source": "action",
                                })

                    # Do not fall back to hands-off while a non-safe action is present.
                    self.frame_count += 1
                    output_frame = self._annotate(frame, w, h)
                    return {"frame": output_frame, "events": events}
                distraction_output, warning_type = hand_alert_pending
                fired = self._trigger_alert(
                    distraction_output,
                    warning_type=warning_type,
                    timestamp_sec=time_sec,
                    trigger_source="hands_off",
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

    def _trigger_alert(self, distraction_output, warning_type: str, timestamp_sec: float, trigger_source: str):
        with self.alert_lock:
            print(f"[DEBUG] _trigger_alert called, audio_playing={self.audio_playing}, last_alert_time={self.last_alert_time:.2f}, timestamp={timestamp_sec:.2f}")
            if timestamp_sec - self.last_alert_time < 5:
                print(f"[BLOCKED] Cooldown active")
                return False
            with self.audio_playing_lock:
                if self.audio_playing:
                    print("[BLOCKED] Audio still playing")
                    return False
                self.audio_playing = True
            self.last_alert_time = timestamp_sec
            self.active_alert_threads += 1

        print(f"[DEBUG] Spawning alert thread")
        threading.Thread(
            target=self._fire_alert,
            args=(distraction_output, warning_type, timestamp_sec, self.play_audio, trigger_source),
            daemon=False
        ).start()
        return True

    def _fire_alert(self, distraction_output, warning_type: str, video_timestamp, play_audio, trigger_source: str):
        try:
            audio_bytes, warning_text = generate_safety_alert_all_groq(
                distraction_output, warning_type=warning_type, play_audio=play_audio
            )
            if isinstance(audio_bytes, bytes) and audio_bytes:
                with self.alert_log_lock:
                    self.alert_log.append((video_timestamp, audio_bytes))
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
            # Clear playing flag so next alert can fire
            with self.audio_playing_lock:
                self.audio_playing = False
            with self.alert_lock:
                self.active_alert_threads -= 1

    def _annotate(self, frame, w, h):
        out = frame.copy()

        # -- Mediapipe overlays (landmarks, pose, steering box, hand labels) --
        if self.mediapipe_enabled and self.last_detection_result is not None:
            r = self.last_detection_result
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            # draw_landmarks_on_image accepts RGB but returns BGR internally
            out = draw_landmarks_on_image(rgb, r["hand_result"])
            # draw_pose_markers expects BGR and returns BGR
            out = draw_pose_markers(out, r["pose_result"], w, h)

            if r["steering_box"]:
                x1, y1, x2, y2 = r["steering_box"]
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(out, "Steering Wheel", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                # Hand labels inside the steering wheel box
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
                # No steering box — draw hand labels at default corner
                label_y = 30
                for side, key in [("Left Hand", "left"), ("Right Hand", "right")]:
                    is_on = self.confirmed_hand_state[key]
                    color = (0, 255, 0) if is_on else (0, 0, 255)
                    text = f"{side}: {'ON' if is_on else 'OFF'}"
                    cv2.putText(out, text, (10, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                    label_y += 24

        # -- Action recognition overlay (annotate_video.py draw_overlay style) --
        if self.videomae_enabled and self.latest_action_result:
            ar = self.latest_action_result
            label = ar["predicted_class"].replace("_", " ").title()
            confidence = ar["confidence"]
            top3 = ar.get("top_k", [])

            overlay = out.copy()
            box_h = 90
            # Semi-transparent dark box at bottom
            cv2.rectangle(overlay, (0, h - box_h), (w, h), (20, 20, 20), -1)
            out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)

            # Main label
            cv2.putText(out, label, (12, h - box_h + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Confidence percentage
            conf_text = f"{confidence * 100:.1f}%"
            cv2.putText(out, conf_text, (12, h - box_h + 58),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 230, 100), 1, cv2.LINE_AA)

            # Top-3 mini bar chart (right side)
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


# ── upload route ───────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
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
    """
    Runs the full pipeline in a background thread.
    Pushes JPEG bytes into job['frame_queue'] and alert dicts into job['event_queue']
    so the WebSocket endpoint can forward them to the browser without re-processing.
    """
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

            # Write to disk (silent)
            out_writer.write(annotated)

            # Push JPEG to WS consumers — drop frame if queue is full (don't block pipeline)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            try:
                fq.put_nowait(buf.tobytes())
            except queue.Full:
                pass

            # Push alert events
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

        # wait for in-flight audio threads
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
        # Signal end of stream to any connected WS clients
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
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Don't expose queues or internal objects to the client
    return {k: v for k, v in job.items() if k not in ("frame_queue", "event_queue", "detection_state")} | {"job_id": job_id}


@app.get("/output/{job_id}")
def download_output(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "Output not ready")
    return FileResponse(job["output_path"], media_type="video/mp4", filename=f"annotated_{job_id}.mp4")


# ── WebSocket — upload job streaming ──────────────────────────────────────────
@app.websocket("/stream/upload/{job_id}")
async def stream_upload_frames(websocket: WebSocket, job_id: str):
    """
    Forwards frames and events produced by the pipeline background thread to the browser.
    No separate video processing here — just queue consumption.
    """
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

    # Background task to listen for toggle messages without blocking the frame loop
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
            # Drain pending alert events first (non-blocking)
            while True:
                try:
                    event = eq.get_nowait()
                    if event is None:
                        break
                    await websocket.send_json(event)
                except queue.Empty:
                    break

            # Get next frame (blocking, but run in executor to not block event loop)
            try:
                frame_bytes = await asyncio.wait_for(
                    loop.run_in_executor(None, fq.get),
                    timeout=600.0,
                )
            except asyncio.TimeoutError:
                break

            if frame_bytes is None:
                # Pipeline finished
                await websocket.send_json({"type": "done"})
                break

            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        pass
    finally:
        listener.cancel()


# ── WebSocket — live webcam ────────────────────────────────────────────────────
@app.websocket("/stream/webcam")
async def stream_webcam(websocket: WebSocket, cam_index: int = 0):
    await websocket.accept()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"Cannot open camera {cam_index}"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    state = DetectionState(fps=fps, play_audio=True)

    # Background task to listen for toggle messages without blocking the frame loop
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
