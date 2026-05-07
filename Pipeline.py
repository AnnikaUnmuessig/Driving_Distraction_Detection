#This is the final pipeline
import cv2
import time
from Steering_wheel_detector import detect_steering_and_hands, draw_landmarks_on_image,draw_pose_markers
from Feedback import generate_safety_alert_all_groq

#Fixed variables
HANDS_OFF_THRESHOLD = 5        # seconds
WHEEL_DETECTION_INTERVAL = 1   # seconds
TIMESFORMER_WINDOW_SIZE = 16   # number of frames for action classification
ACTION_OVERLAP = 8             # start new action classification every 8 frames
DEBOUNCE_THRESHOLD = 3          # number of consecutive detections to confirm state change

#Dummy action classifier 
def classify_action(frames_buffer):
    #placeholder
    return None


def run_pipeline(video_path=None):
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up VideoWriter (only when processing a file)
    out = None
    if video_path:
        output_path = video_path.rsplit(".", 1)[0] + "_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        print(f"Saving annotated video to: {output_path}")

    # State tracking
    hands_off_since = None
    last_wheel_check_time = 0
    frames_buffer = []
    frame_count = 0
    last_action_frame = -ACTION_OVERLAP
    last_annotated_frame = None   # Cache last annotated frame between wheel checks
    last_status_text = []         # Cache status overlays between checks

    # Debounce state
    confirmed_hand_state = {"left": True, "right": True}
    pending_hand_state = None
    pending_count = 0

    print("Pipeline running. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        frame_count += 1

        # ── 1. STEERING WHEEL + HAND DETECTION ──
        if current_time - last_wheel_check_time >= WHEEL_DETECTION_INTERVAL:
            last_wheel_check_time = current_time

            temp_path = "_temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            result = detect_steering_and_hands(temp_path)
            mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_annotated_frame = draw_landmarks_on_image(mp_frame, result["hand_result"])
            last_annotated_frame = draw_pose_markers(last_annotated_frame, result["pose_result"], frame_w, frame_h)

            if result["steering_box"]:
                x1, y1, x2, y2 = result["steering_box"]
                cv2.rectangle(last_annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 8)
                cv2.putText(last_annotated_frame, "Steering Wheel", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            new_state = {
                "left": result["left_hand_on"],
                "right": result["right_hand_on"]
            }

            # ── Debounce logic ──
            if new_state == confirmed_hand_state:
                # Result matches confirmed state — reset any pending change
                pending_hand_state = None
                pending_count = 0
            elif new_state == pending_hand_state:
                # Same change seen again — increment counter
                pending_count += 1
                if pending_count >= DEBOUNCE_THRESHOLD:
                    # Change confirmed, commit it
                    print(f"State change confirmed: {confirmed_hand_state} → {new_state}")
                    confirmed_hand_state = new_state
                    pending_hand_state = None
                    pending_count = 0
            else:
                # New candidate change — start tracking it
                pending_hand_state = new_state
                pending_count = 1

            # ── Use confirmed_hand_state for alert logic (not raw result) ──
            hands_on_wheel = confirmed_hand_state["left"] or confirmed_hand_state["right"]

            if hands_on_wheel:
                hands_off_since = None
            else:
                if hands_off_since is None:
                    hands_off_since = current_time

                hands_off_duration = current_time - hands_off_since
                print(f"Hands off wheel for {hands_off_duration:.1f}s")

                if hands_off_duration >= HANDS_OFF_THRESHOLD:
                    print("⚠️  ALERT: Hands off wheel too long!")
                    distraction_output = {
                        "distracted": "yes",
                        "distraction_type": "hands off wheel",
                        "type of warning": "mid-heavy"
                    }
                    generate_safety_alert_all_groq(distraction_output)
                    hands_off_since = current_time
                    last_status_text.append(("⚠ HANDS OFF WHEEL", (0, 0, 255)))
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
                distraction_output = {
                    "distracted": "yes",
                    "distraction_type": action,
                    "type of warning": "light-mid"
                }
                generate_safety_alert_all_groq(distraction_output)
                last_status_text.append((f"⚠ {action.upper()}", (0, 165, 255)))

        # ── 3. COMPOSE OUTPUT FRAME ──
        # Use the last annotated frame (with landmarks/boxes) as the base,
        # falling back to the raw frame before the first wheel check fires
        output_frame = last_annotated_frame if last_annotated_frame is not None else frame.copy()

        # Always stamp confirmed hand state on every frame
        state_y = 100
        for side, key in [("LEFT", "left"), ("RIGHT", "right")]:
            is_on = confirmed_hand_state[key]
            text = f"{side} HAND: {'ON' if is_on else 'OFF'}"
            color = (0, 255, 0) if is_on else (0, 0, 255)
            cv2.putText(output_frame, text, (50, state_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
            state_y += 100

        # Pending debounce indicator (remove once tuning is done)
        if pending_hand_state is not None:
            pending_text = f"pending ({pending_count}/{DEBOUNCE_THRESHOLD}): L={'ON' if pending_hand_state['left'] else 'OFF'} R={'ON' if pending_hand_state['right'] else 'OFF'}"
            cv2.putText(output_frame, pending_text, (50, state_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        # Stamp any active alert text at the bottom
        for i, (text, color) in enumerate(last_status_text):
            cv2.putText(output_frame, text, (50, frame_h - 60 - i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)

        if out:
            out.write(output_frame)

        # Still show preview while processing (remove if you want headless)
        cv2.imshow("Driver Monitor", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
        print(f"✅ Done. Annotated video saved to: {output_path}")
    cv2.destroyAllWindows()

run_pipeline("test_data/test_video2.mp4")