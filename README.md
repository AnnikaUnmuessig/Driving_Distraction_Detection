# Driver Monitor — Full Stack Setup

## Architecture

```
browser (React + Vite)
  ├── UploadSource   ──POST /upload──────────────► FastAPI
  │                  ──WS  /stream/upload/{id}───► frame stream
  │                  ──GET /output/{id}──────────► _final.mp4
  │
  └── WebcamSource   ──WS  /stream/webcam─────────► frame stream
                                                    cv2.VideoCapture(cam_index)
```

### Upload flow
1. User drops a video file → `POST /upload` → backend saves file, spawns background thread
2. Thread runs `DetectionState.process_frame()` on every frame, writes silent MP4
3. After processing: waits for alert audio threads, calls `build_audio_track` + `mux_audio_into_video`
4. Frontend polls `GET /jobs/{job_id}` every second for progress
5. WS `/stream/upload/{job_id}` sends JPEG frames in real time during processing
6. On completion: download button hits `GET /output/{job_id}`

### Webcam flow
1. User picks camera index and clicks Start
2. Frontend opens `WS /stream/webcam?cam_index=N`
3. Backend opens `cv2.VideoCapture(N)`, runs pipeline loop, sends JPEG frames + alert JSON
4. Frontend renders frames as `<img>` via Blob URLs (old URLs revoked to avoid memory leaks)
5. Click Stop → WebSocket closed → `cap.release()` called server-side

### Message protocol (WebSocket)
- **Binary frames**: raw JPEG bytes — render directly as `<img src={blobUrl}>`
- **JSON messages**:
  ```json
  { "type": "alert", "distraction_type": "hands off wheel", "severity": "mid-heavy" }
  { "type": "done" }
  { "type": "error", "message": "Cannot open camera 0" }
  ```

---

## Backend setup

```bash
cd backend

# Install dependencies (add your existing packages too)
pip install -r requirements.txt


# Copy your existing pipeline modules into backend/
# Required: Steering_wheel_detector.py, Feedback.py, pipeline.py
# pipeline.py must export: build_audio_track, classify_action, mux_audio_into_video,
#   ACTION_OVERLAP, DEBOUNCE_THRESHOLD, HANDS_OFF_THRESHOLD,
#   TIMESFORMER_WINDOW_SIZE, WHEEL_DETECTION_INTERVAL

# Run
uvicorn main:app --reload --port 8000
```

Your existing `pipeline.py` constants and helpers are imported directly — no changes needed.
Just make sure `pipeline.py` is in the same directory as `main.py`, or installed as a package.

---

## Frontend setup

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173
```

---

## Project structure

```
driver-monitor/
├── backend/
│   ├── main.py              ← FastAPI app (this file)
│   ├── requirements.txt
│   ├── pipeline.py          ← your existing pipeline (copy here)
│   ├── Steering_wheel_detector.py
│   ├── Feedback.py
│   ├── uploads/             ← auto-created
│   └── outputs/             ← auto-created
│
└── frontend/
    ├── package.json
    └── src/
        ├── App.tsx                      ← root layout
        ├── components/
        │   ├── VideoPanel.tsx           ← shared frame view + alert feed
        │   ├── UploadSource.tsx         ← file upload card
        │   └── WebcamSource.tsx         ← live webcam card
        └── hooks/
            ├── useVideoStream.ts        ← WebSocket frame/event consumer
            └── useUploadJob.ts          ← upload XHR + job polling
```

---

## Running both sources simultaneously

Both sources run entirely independently — separate `DetectionState` instances,
separate `cv2.VideoCapture` handles, separate WebSocket connections.
There is no shared state between them.

To run both at once: start both panels in the UI. The backend handles them
in concurrent async tasks (upload in a thread pool, webcam in async tasks).

---

## Adapting pipeline.py for import

Extract the constants and helpers from your existing `pipeline.py` so they can be imported:

```python
# pipeline.py — make sure these are importable at module level
HANDS_OFF_THRESHOLD = 1
WHEEL_DETECTION_INTERVAL = 1
TIMESFORMER_WINDOW_SIZE = 16
ACTION_OVERLAP = 8
DEBOUNCE_THRESHOLD = 3

def classify_action(frames_buffer): ...
def build_audio_track(alert_log, total_duration_seconds): ...
def mux_audio_into_video(silent_video_path, final_output_path, pcm_bytes, total_duration_seconds): ...
```

Guard the `run_pipeline(...)` call at the bottom with `if __name__ == "__main__":` so it
doesn't execute on import.
