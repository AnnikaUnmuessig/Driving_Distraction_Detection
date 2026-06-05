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
  { "type": "alert", "distraction_type": "texting_right", "severity": "heavy", "trigger_source": "action" }
  { "type": "alert_text", "text": "...", "distraction_type": "texting_right", "severity": "heavy", "trigger_source": "action" }
  { "type": "hand_state", "confirmed": { "left": true, "right": false }, "pending_count": 0 }
  { "type": "done" }
  { "type": "error", "message": "Cannot open camera 0" }
  ```
- **Client → server toggles**: `{ "type": "toggle", "target": "mediapipe"|"videomae"|"audio", "enabled": true }`

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
# Required: Steering_wheel_detector.py, Feedback.py, pipeline.py, action_recognition.py
# pipeline.py exports (used by main.py): build_audio_track, mux_audio_into_video,
#   ActionRecognitionWorker, ACTION_CONFIDENCE_THRESHOLD, get_action_warning_type

# Set environment variables (create backend/.env)
# GROQ_API_KEY — LLM voice alert text; ROBOFLOW_API — steering wheel detection
echo "GROQ_API_KEY=your_key_here" > .env
echo "ROBOFLOW_API=your_key_here" >> .env

# Run
uvicorn main:app --reload --port 8000
```

`main.py` owns live detection timing and alert rules in `DetectionState`.
`pipeline.py` still provides audio muxing, `ActionRecognitionWorker`, and severity mapping;
its standalone CLI script uses older constants (see below).

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

## Detection & alert tuning

The **FastAPI app** (`main.py` → `DetectionState`) is what upload and webcam use. Tune these in `DetectionState.__init__`:

| Setting | Default | Role |
|--------|---------|------|
| `action_interval_sec` | `1.0` | Min seconds between VideoMAE inference calls (set in `DetectionState.__init__`) |
| `action_persist_sec` | `4.0` | Seconds a warning-worthy action must persist before an action alert fires |
| `ACTION_CONFIDENCE_THRESHOLD` | `0.5` | Imported from `pipeline.py`; min smoothed confidence for action tracking |
| `debounce_on_frames` | `3` | Consecutive frames to confirm hand **on** wheel |
| `debounce_off_frames` | `15` | Consecutive frames to confirm hand **off** wheel |
| `hands_off_one_hand_threshold` | `5.0` | Seconds with one hand off before hands-off alert |
| `hands_off_both_hands_threshold` | `2.0` | Seconds with both hands off before hands-off alert |
| `same_action_cooldown` | `5.0` | Wall-clock seconds between alerts for the **same** distraction |
| `diff_action_cooldown` | `2.0` | Wall-clock seconds between alerts for **different** distractions |
| `ema_alpha` | `0.4` | EMA smoothing weight for action class probabilities |
| Roboflow re-detect interval | `30.0` s | Steering wheel box refresh (`last_roboflow_time` in `process_frame`) |
| VideoMAE window size | `16` frames | Hard-coded sample count in `process_frame` |

**Alert logic (upload & webcam):**

- **Action alert**: warning-worthy class above confidence, persisted for `action_persist_sec`.
- **Hands-off alert**: only if current action is `safe_driving` (or no recent action); suppressed while a warning-worthy action is being tracked.
- **Voice toggle**: `voice_alerts_enabled` + `play_audio` only (does not disable VideoMAE).
- **Upload timing**: `use_video_timeline=True` so detection uses video position, independent of speaker playback.

### `pipeline.py` (legacy CLI script only)

These module-level constants still exist for the standalone `pipeline.py` video script, **not** for `main.py`:

```python
"""
HANDS_OFF_THRESHOLD = 1
WHEEL_DETECTION_INTERVAL = 1
VIDEOMAE_WINDOW_SIZE = 16
ACTION_OVERLAP = 30
DEBOUNCE_THRESHOLD = 3
ACTION_CONFIDENCE_THRESHOLD = 0.5
"""
```

Shared helpers imported by `main.py`:

```python
def build_audio_track(alert_log, total_duration_seconds): ...
def mux_audio_into_video(silent_video_path, final_output_path, pcm_bytes, total_duration_seconds): ...
def get_action_warning_type(predicted_class: str) -> str | None: ...
class ActionRecognitionWorker: ...
```

