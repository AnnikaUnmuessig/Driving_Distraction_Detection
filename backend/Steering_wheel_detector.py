"""
This module detects the steering wheel and the driver's hands.
It uses Roboflow for steering wheel detection and MediaPipe for hand/pose tracking.
"""

from roboflow import Roboflow
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import tempfile
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

project_root = os.path.dirname(script_dir)
models_dir = os.path.join(project_root, 'models')

rf_api_key = os.environ.get("ROBOFLOW_API")
rf = Roboflow(api_key=rf_api_key)

project = rf.workspace().project("steering-detection-s2wqt")
model = project.version(1).model

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=os.path.join(models_dir, 'hand_landmarker.task')),
    num_hands=2, min_hand_detection_confidence=0.3
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=os.path.join(models_dir, 'pose_landmarker_lite.task')),
    min_pose_detection_confidence=0.4
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


def draw_landmarks_on_image(rgb_image, hand_results):
    """Draws hand landmarks on the image using the MediaPipe HandLandmarker result."""
    annotated_image = cv2.cvtColor(np.copy(rgb_image), cv2.COLOR_RGB2BGR)
    if not hand_results.hand_landmarks:
        return annotated_image

    hand_landmark_style = drawing_styles.get_default_hand_landmarks_style()
    hand_connection_style = drawing_styles.get_default_hand_connections_style()

    for hand_landmarks in hand_results.hand_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=hand_landmark_style,
            connection_drawing_spec=hand_connection_style)
    return annotated_image

def draw_pose_markers(bgr_image, pose_results, img_w, img_h):
    """Draws pose landmark markers on the image (disabled)."""
    return bgr_image


def detect_steering_and_hands(frame_or_path, steering_box=None):
    """Detects steering wheel box and hand positions relative to the wheel using Roboflow and MediaPipe."""
    if isinstance(frame_or_path, str):
        bgr_frame = cv2.imread(frame_or_path)
        roboflow_source = frame_or_path
        _delete_temp = False
    else:
        bgr_frame = frame_or_path
        roboflow_source = None
        _delete_temp = False

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = rgb_frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    x1 = y1 = x2 = y2 = 0
    if steering_box is None:
        if roboflow_source is None:
            _, encoded = cv2.imencode(".jpg", bgr_frame)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(encoded.tobytes())
            tmp.flush()
            tmp.close()
            roboflow_source = tmp.name
            _delete_temp = True

        try:
            output = model.predict(roboflow_source, confidence=40).json()
        finally:
            if _delete_temp:
                os.unlink(roboflow_source)

        if output['predictions']:
            pred = output['predictions'][0]
            x_center, y_center, w, h = pred['x'], pred['y'], pred['width'], pred['height']
            x1, y1, x2, y2 = (int(x_center - w/2), int(y_center - h/2),
                              int(x_center + w/2), int(y_center + h/2))
            steering_box = (x1, y1, x2, y2)
    else:
        x1, y1, x2, y2 = steering_box

    hand_result = hand_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)
    detected_hands = []
    left_on = False
    right_on = False

    if hand_result.hand_landmarks:
        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            score = hand_result.handedness[idx][0].score
            raw_label = hand_result.handedness[idx][0].category_name
            wrist_y = hand_lms[0].y * img_h

            is_inside = False
            if steering_box:
                landmarks_inside = sum(
                    1 for lm in hand_lms
                    if x1 <= (lm.x * img_w) <= x2 and y1 <= (lm.y * img_h) <= y2
                )
                is_inside = landmarks_inside >= 2

            detected_hands.append({
                "hand_lms": hand_lms,
                "raw_label": raw_label,
                "score": score,
                "wrist_y": wrist_y,
                "is_inside": is_inside
            })

    if len(detected_hands) == 2:
        labels = [h["raw_label"] for h in detected_hands]
        scores = [h["score"] for h in detected_hands]

        if labels[0] != labels[1] and min(scores) > 0.75:
            for hand in detected_hands:
                hand["label"] = "Right" if hand["raw_label"] == "Right" else "Left"
        else:
            detected_hands.sort(key=lambda h: h["wrist_y"])
            detected_hands[0]["label"] = "Left"
            detected_hands[1]["label"] = "Right"

    elif len(detected_hands) == 1:
        hand = detected_hands[0]
        hand["label"] = "Right" if hand["raw_label"] == "Right" else "Left"

        if hand["label"] == "Left":
            left_on = hand["is_inside"]
            right_on = False
        else:
            right_on = hand["is_inside"]
            left_on = False

    for hand in detected_hands:
        if hand.get("is_inside") and "label" in hand:
            if hand["label"] == "Left": left_on = True
            elif hand["label"] == "Right": right_on = True

    POSE_WRIST_INDICES = {"Left": 15, "Right": 16}
    POSE_VISIBILITY_THRESHOLD = 0.4

    if steering_box and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]

        for side, current_status in [("Left", left_on), ("Right", right_on)]:
            if current_status:
                continue

            side_detected = any(h.get("label") == side for h in detected_hands)
            if side_detected:
                continue

            wrist_lm = lms[POSE_WRIST_INDICES[side]]
            if wrist_lm.visibility < POSE_VISIBILITY_THRESHOLD:
                print(f"[Fallback] {side} wrist pose landmark not visible enough, skipping.")
                continue

            wrist_px = wrist_lm.x * img_w
            wrist_py = wrist_lm.y * img_h
            wrist_in_box = x1 <= wrist_px <= x2 and y1 <= wrist_py <= y2

            print(f"[Fallback] {side} hand not detected by MediaPipe. "
                  f"Pose wrist at ({wrist_px:.0f}, {wrist_py:.0f}), "
                  f"in box: {wrist_in_box}")

            if side == "Left":
                left_on = wrist_in_box
            else:
                right_on = wrist_in_box

    return {
        "steering_box": steering_box,
        "left_hand_on": left_on,
        "right_hand_on": right_on,
        "hands_on_wheel": left_on or right_on,
        "hand_result": hand_result,
        "pose_result": pose_result,
    }