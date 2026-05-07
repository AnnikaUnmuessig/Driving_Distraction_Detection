#SET UP: IMPORT LIBRARIES; API KEYS; HELPER FUNCTIONS
from roboflow import Roboflow
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import time
from dotenv import load_dotenv

load_dotenv()

rf_api_key = os.environ.get("ROBOFLOW_API")
rf = Roboflow(api_key=rf_api_key)

# Steering Detection Model
project = rf.workspace().project("steering-detection-s2wqt")
model = project.version(1).model

# Hand Landmarker
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='./models/hand_landmarker.task'),
    num_hands=2, min_hand_detection_confidence=0.3
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Pose Landmarker
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='./models/pose_landmarker_lite.task'),
    min_pose_detection_confidence=0.4
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)


def draw_landmarks_on_image(rgb_image, hand_results):
    """Draws hand landmarks using the Tasks API result."""
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
    """Draws wrist/elbow markers using the Tasks API result."""
    if not pose_results.pose_landmarks:
        return bgr_image

    # Use the first person detected
    lms = pose_results.pose_landmarks[0]
    
    # Indices: 15=L_Wrist, 16=R_Wrist, 13=L_Elbow, 14=R_Elbow
    keypoints = {"L Wrist": lms[15], "R Wrist": lms[16], "L Elbow": lms[13], "R Elbow": lms[14]}

    for name, lm in keypoints.items():
        if lm.visibility < 0.4: continue
        px, py = int(lm.x * img_w), int(lm.y * img_h)
        # Orange for Left, Blue for Right
        color = (255, 128, 0) if "L" in name else (0, 128, 255)
        shape = cv2.MARKER_CROSS if "Elbow" in name else cv2.MARKER_STAR
        cv2.drawMarker(bgr_image, (px, py), color, markerType=shape, markerSize=25, thickness=2)
        cv2.putText(bgr_image, name, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return bgr_image

def detect_steering_and_hands(image_path):
    # Load and preprocess image
    mp_image = mp.Image.create_from_file(image_path)
    rgb_frame = mp_image.numpy_view()
    img_h, img_w = rgb_frame.shape[:2]

    # 1. Detect Steering Wheel (YOLO/Roboflow)
    output = model.predict(image_path, confidence=40).json()
    steering_box = None
    if output['predictions']:
        pred = output['predictions'][0]
        x_center, y_center, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        x1, y1, x2, y2 = (int(x_center - w/2), int(y_center - h/2), 
                          int(x_center + w/2), int(y_center + h/2))
        steering_box = (x1, y1, x2, y2)

    # 2. Detect Hands and Pose (MediaPipe)
    hand_result = hand_detector.detect(mp_image)
    pose_result = pose_detector.detect(mp_image)
    print(f"Hand result: {hand_result}")
    detected_hands = []
    left_on = False
    right_on = False

    # Pre-process detections for logic checks
    if hand_result.hand_landmarks:
        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            score = hand_result.handedness[idx][0].score
            raw_label = hand_result.handedness[idx][0].category_name
            wrist_x = hand_lms[0].x * img_w
            wrist_y = hand_lms[0].y * img_h

            is_inside = False
            if steering_box:
                landmarks_inside = sum(
                    1 for lm in hand_lms
                    if x1 <= (lm.x * img_w) <= x2 and y1 <= (lm.y * img_h) <= y2
                )
                is_inside = landmarks_inside >= 2 # At least 5 landmarks inside to consider "on wheel"

            detected_hands.append({
                "hand_lms": hand_lms,
                "raw_label": raw_label,
                "score": score,
                "wrist_y": wrist_y,
                "is_inside": is_inside
            })

    # 3. Custom Logic: Evaluate Left/Right Status
    if len(detected_hands) == 2:
        labels = [h["raw_label"] for h in detected_hands]
        scores = [h["score"] for h in detected_hands]

        if labels[0] != labels[1] and min(scores) > 0.75:
            # High confidence distinct labels -> apply mirroring fix (swap)
            for hand in detected_hands:
                hand["label"] = "Right" if hand["raw_label"] == "Right" else "Left"
        else:
            # Conflicting or low confidence -> use geometry (Vertical position on wheel)
            detected_hands.sort(key=lambda h: h["wrist_y"])
            detected_hands[0]["label"] = "Left"   # Physically higher in frame
            detected_hands[1]["label"] = "Right"  # Physically lower in frame

    elif len(detected_hands) == 1:
            hand = detected_hands[0]
            # Always flip the label (to fix the mirroring) 
            # but only apply the 'ON' status to that specific hand
            hand["label"] = "Right" if hand["raw_label"] == "Right" else "Left"
            
            # We explicitly ensure the OTHER hand is False
            if hand["label"] == "Left":
                left_on = hand["is_inside"]
                right_on = False
            else:
                right_on = hand["is_inside"]
                left_on = False

    # Final assignment based on processed labels
    for hand in detected_hands:
        if hand.get("is_inside") and "label" in hand:
            if hand["label"] == "Left": left_on = True
            elif hand["label"] == "Right": right_on = True

    """
    # 4. Visualization
    annotated_image = draw_landmarks_on_image(rgb_frame, hand_result)
    annotated_image = draw_pose_markers(annotated_image, pose_result, img_w, img_h)

    if steering_box:
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    status_y = 100
    for side, is_on in [("LEFT", left_on), ("RIGHT", right_on)]:
        text = f"{side} HAND: {'ON' if is_on else 'OFF'}"
        color = (0, 255, 0) if is_on else (0, 0, 255)
        cv2.putText(annotated_image, text, (50, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        status_y += 100

    plt.figure(figsize=(15, 10))
    plt.imshow(annotated_image[:, :, ::-1])
    plt.axis('off')
    #plt.show()
    """
    return {
        "steering_box": steering_box,
        "left_hand_on": left_on,
        "right_hand_on": right_on,
        "hands_on_wheel": left_on or right_on,
        "hand_result": hand_result,
        "pose_result": pose_result,
    }
"""
detection_result = detect_steering_and_hands("notebooks/test_img2.png")
print("Detection Result:", detection_result)
"""