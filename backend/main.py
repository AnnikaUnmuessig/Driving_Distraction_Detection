"""
Driver Monitor — FastAPI backend
Exposes:
  POST /upload                      — accepts video, runs pipeline in background thread
  GET  /stream/upload/{job_id}      — WebSocket, streams frames produced by the pipeline thread
  GET  /stream/webcam               — WebSocket, runs pipeline on cv2.VideoCapture(cam_index)
  GET  /jobs/{job_id}               — poll status + progress
  GET  /output/{job_id}             — download the _final.mp4 once ready
"""

import asyncio
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
    ACTION_OVERLAP,
    DEBOUNCE_THRESHOLD,
    HANDS_OFF_THRESHOLD,
    TIMESFORMER_WINDOW_SIZE,
    WHEEL_DETECTION_INTERVAL,
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
        "error": None,
        "frame_queue": queue.Queue(maxsize=120),   # ~4 s buffer at 30 fps
        "event_queue": queue.Queue(maxsize=200),
    }


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


# ── shared detection state ─────────────────────────────────────────────────────
class DetectionState:
    def __init__(self):
        self.hands_off_since: Optional[float] = None
        self.last_wheel_check_time: float = 0
        self.frames_buffer: list = []
        self.frame_count: int = 0
        self.last_action_frame: int = -ACTION_OVERLAP
        self.last_detection_result: Optional[dict] = None
        self.confirmed_hand_state: dict = {"left": True, "right": True}
        self.pending_hand_state: Optional[dict] = None
        self.pending_count: int = 0
        self.alert_active: bool = False
        self.alert_lock = threading.Lock()
        self.alert_log: list = []
        self.alert_log_lock = threading.Lock()

    def process_frame(self, frame: np.ndarray, current_time: float, cap_msec: float):
        h, w = frame.shape[:2]
        events = []

        if current_time - self.last_wheel_check_time >= WHEEL_DETECTION_INTERVAL:
            self.last_wheel_check_time = current_time
            result = detect_steering_and_hands(frame)

            if result["steering_box"] is None and self.last_detection_result is not None:
                if self.last_detection_result.get("steering_box") is not None:
                    result["steering_box"] = self.last_detection_result["steering_box"]

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
                    # Send hand state update if it changed
                    if old_state != self.confirmed_hand_state:
                        events.append({
                            "type": "hand_state",
                            "confirmed": self.confirmed_hand_state,
                            "pending_count": self.pending_count
                        })
            else:
                self.pending_hand_state = new_state
                self.pending_count = 1
                # Send pending state update
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
                self.hands_off_since = current_time
            off_duration = current_time - self.hands_off_since
            if off_duration >= HANDS_OFF_THRESHOLD:
                fired = self._trigger_alert(
                    {"distracted": "yes", "distraction_type": "hands off wheel", "type of warning": "mid-heavy"},
                    cap_msec / 1000.0,
                )
                if fired:
                    self.hands_off_since = current_time
                    events.append({"type": "alert", "distraction_type": "hands off wheel", "severity": "mid-heavy"})

        self.frames_buffer.append(frame)
        if len(self.frames_buffer) > TIMESFORMER_WINDOW_SIZE:
            self.frames_buffer.pop(0)

        if (
            len(self.frames_buffer) == TIMESFORMER_WINDOW_SIZE
            and self.frame_count - self.last_action_frame >= ACTION_OVERLAP
        ):
            self.last_action_frame = self.frame_count
            action = classify_action(self.frames_buffer)
            if action:
                self._trigger_alert(
                    {"distracted": "yes", "distraction_type": action, "type of warning": "light-mid"},
                    cap_msec / 1000.0,
                )
                events.append({"type": "alert", "distraction_type": action, "severity": "light-mid"})

        self.frame_count += 1
        output_frame = self._annotate(frame, w, h)
        return {"frame": output_frame, "events": events}

    def _trigger_alert(self, distraction_output, video_timestamp):
        with self.alert_lock:
            if self.alert_active:
                return False
            self.alert_active = True
        threading.Thread(target=self._fire_alert, args=(distraction_output, video_timestamp), daemon=True).start()
        return True

    def _fire_alert(self, distraction_output, video_timestamp):
        try:
            audio_bytes = generate_safety_alert_all_groq(distraction_output)
            if isinstance(audio_bytes, bytes) and audio_bytes:
                with self.alert_log_lock:
                    self.alert_log.append((video_timestamp, audio_bytes))
        except Exception as e:
            print(f"Alert error: {e}")
        finally:
            with self.alert_lock:
                self.alert_active = False

    def _annotate(self, frame, w, h):
        if self.last_detection_result is None:
            return frame.copy()
        r = self.last_detection_result
        mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = draw_landmarks_on_image(mp_frame, r["hand_result"])
        out = draw_pose_markers(out, r["pose_result"], w, h)
        if r["steering_box"]:
            x1, y1, x2, y2 = r["steering_box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 8)
            cv2.putText(out, "Steering Wheel", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        y = 100
        for side, key in [("LEFT", "left"), ("RIGHT", "right")]:
            is_on = self.confirmed_hand_state[key]
            cv2.putText(out, f"{side}: {'ON' if is_on else 'OFF'}", (50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0) if is_on else (0, 0, 255), 6)
            y += 100
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
        state = DetectionState()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _update_job(job_id, status="error", error="Could not open video file")
            fq.put(None)
            eq.put(None)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        silent_path = str(OUTPUT_DIR / f"{job_id}_silent.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(silent_path, fourcc, fps, (w, h))

        frame_count = 0
        while cap.isOpened():
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
                still = state.alert_active
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
                os.rename(silent_path, final_path)
        else:
            os.rename(silent_path, final_path)

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
    # Don't expose queues to the client
    return {k: v for k, v in job.items() if k not in ("frame_queue", "event_queue")} | {"job_id": job_id}


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
                    timeout=30.0,
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


# ── WebSocket — live webcam ────────────────────────────────────────────────────
@app.websocket("/stream/webcam")
async def stream_webcam(websocket: WebSocket, cam_index: int = 0):
    await websocket.accept()
    state = DetectionState()

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        await websocket.send_json({"type": "error", "message": f"Cannot open camera {cam_index}"})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps

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
        cap.release()
