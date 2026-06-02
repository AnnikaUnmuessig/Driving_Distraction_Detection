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

### Detection Pipeline
The backend uses a multi-modal approach for comprehensive distraction detection:

- **Hand Detection**: MediaPipe HandLandmarker (2 hands, confidence ≥ 0.3)
- **Steering Wheel Detection**: Roboflow steering wheel detector (confidence ≥ 0.4)
- **Pose Fallback**: MediaPipe PoseLandmarker (wrist detection when hands not visible)
- **Action Classification**: VideoMAE model for temporal action/distraction recognition
- **Alert Generation**: LLM-powered voice alerts via Groq API

### Upload flow
1. User drops a video file → `POST /upload` → backend saves file, spawns background thread
2. Thread runs `DetectionState.process_frame()` on every frame:
   - Detects hands on steering wheel (MediaPipe + fallback pose landmarks)
   - Classifies driver actions using VideoMAE on temporal frame windows
   - Triggers safety alerts for dangerous behaviors
   - Writes annotated silent MP4
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
  { "type": "alert", "distraction_type": "texting", "severity": "heavy" }
  { "type": "done" }
  { "type": "error", "message": "Cannot open camera 0" }
  ```

---

## Backend setup

```bash
cd backend

# Install dependencies (add your existing packages too)
pip install -r requirements.txt

# Ensure models directory exists with required MediaPipe models:
# - models/hand_landmarker.task
# - models/pose_landmarker_lite.task
# - models/pose_landmarker_heavy.task
# - models/video_mae/config.json
# - models/video_mae/model.safetensors

# Copy your existing pipeline modules into backend/
# Required: Steering_wheel_detector.py, Feedback.py, pipeline.py
# pipeline.py must export: build_audio_track, classify_action, mux_audio_into_video,
#   ACTION_OVERLAP, DEBOUNCE_THRESHOLD, HANDS_OFF_THRESHOLD,
#   VIDEOMAE_WINDOW_SIZE, WHEEL_DETECTION_INTERVAL, ACTION_CONFIDENCE_THRESHOLD

# Set environment variables (create backend/.env)
echo "ROBOFLOW_API=your_api_key_here" > .env

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
HANDS_OFF_THRESHOLD = 1              # Seconds before "hands off wheel" alert
WHEEL_DETECTION_INTERVAL = 1         # Frames between steering wheel checks
VIDEOMAE_WINDOW_SIZE = 16            # Frames per action classification window
ACTION_OVERLAP = 8                   # Overlap between consecutive windows
DEBOUNCE_THRESHOLD = 3               # Frames to debounce hand state changes
ACTION_CONFIDENCE_THRESHOLD = 0.6    # Min confidence for action alerts

def classify_action(frames_buffer): 
    """Classify driver action/distraction from frame buffer using VideoMAE."""
    ...

def build_audio_track(alert_log, total_duration_seconds): 
    """Generate audio track from alert log."""
    ...

def mux_audio_into_video(silent_video_path, final_output_path, pcm_bytes, total_duration_seconds): 
    """Combine video with audio track."""
    ...

def get_action_warning_type(action):
    """Return severity level ('light', 'mid', 'heavy') for action, or None if safe."""
    ...
```

