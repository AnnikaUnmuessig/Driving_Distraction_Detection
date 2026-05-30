# action_recognition.py
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import VideoMAEImageProcessor, VideoMAEModel
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

NUM_FRAMES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path definitions
models_dir = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = models_dir / "unified.pth"
HAND_LANDMARKER_PATH = models_dir / "hand_landmarker.task"

ID2LABEL = {
    0: "safe_driving",
    1: "texting_right",
    2: "phonecall_right",
    3: "texting_left",
    4: "phonecall_left",
    5: "radio",
    6: "drinking",
    7: "reach_side",
    8: "hair_and_makeup",
    9: "change_gear",
}

class UnifiedDistractionClassifier(nn.Module):
    def __init__(self, model_id, num_classes=10):
        super().__init__()
        # 1. VideoMAE Backbone (Feature Extractor)
        self.videomae = VideoMAEModel.from_pretrained(model_id)
        
        # 2. Landmark Sequence Recurrent Network (LSTM)
        # input_size = 126 (2 hands * 21 keypoints * 3 coordinates)
        self.lstm = nn.LSTM(
            input_size=126,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Fusion Layer & Classification Head
        # VideoMAEPooled: 768-d | Bi-LSTM Hidden: 256-d (128 * 2)
        self.fc = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, pixel_values, landmark_coords):
        # Forward VideoMAE
        # pixel_values shape: (batch, 16, 3, 224, 224)
        outputs = self.videomae(pixel_values)
        # Pool features over patches / tokens: (batch, 768)
        video_features = outputs.last_hidden_state.mean(dim=1)
        
        # Forward LSTM for MediaPipe landmarks
        # landmark_coords shape: (batch, 16, 126)
        lstm_out, _ = self.lstm(landmark_coords)
        # Pool features over sequence steps: (batch, 256)
        landmark_features = lstm_out.mean(dim=1)
        
        # Feature Fusion
        fused = torch.cat([video_features, landmark_features], dim=1)  # (batch, 1024)
        
        # Classification Head
        logits = self.fc(fused)
        return logits


class ActionRecognizer:
    def __init__(self):
        from huggingface_hub import login
        if HF_TOKEN:
            try:
                login(token=HF_TOKEN)
            except Exception as e:
                print(f"[WARN] HF login failed: {e}")
        
        BASE = "MCG-NJU/videomae-base-finetuned-kinetics"
        
        # Load VideoMAE processor
        self.processor = VideoMAEImageProcessor.from_pretrained(BASE, token=HF_TOKEN if HF_TOKEN else None)
        
        # Initialize dedicated MediaPipe hand landmarker
        if not HAND_LANDMARKER_PATH.exists():
            raise FileNotFoundError(f"MediaPipe Hand Landmarker model not found at {HAND_LANDMARKER_PATH}")
            
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(HAND_LANDMARKER_PATH)
        )
        options = mp_tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.4,
        )
        self.hand_detector = mp_tasks.vision.HandLandmarker.create_from_options(options)
        
        # Initialize custom multimodal classifier and load weights
        self.model = UnifiedDistractionClassifier(BASE, num_classes=10)
        
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Unified model checkpoint not found at {MODEL_PATH}")
            
        print(f"Loading weights from {MODEL_PATH}...")
        state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE)
        
        # Adjust state dict keys due to transformers version discrepancies (q_bias/v_bias vs query.bias/value.bias)
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if "attention.attention.q_bias" in k:
                new_key = k.replace("attention.attention.q_bias", "attention.attention.query.bias")
                adjusted_state_dict[new_key] = v
                # Initialize key.bias to zeros of the same shape/type
                key_key = k.replace("attention.attention.q_bias", "attention.attention.key.bias")
                adjusted_state_dict[key_key] = torch.zeros_like(v)
            elif "attention.attention.v_bias" in k:
                new_key = k.replace("attention.attention.v_bias", "attention.attention.value.bias")
                adjusted_state_dict[new_key] = v
            else:
                adjusted_state_dict[k] = v

        self.model.load_state_dict(adjusted_state_dict)
        self.model.eval().to(DEVICE)
        
        self.id2label = ID2LABEL

    def _extract_landmarks(self, frame_rgb) -> np.ndarray:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.hand_detector.detect(mp_image)
        
        left_hand_coords = np.zeros((21, 3), dtype=np.float32)
        right_hand_coords = np.zeros((21, 3), dtype=np.float32)
        
        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                side = handedness[0].category_name  # 'Left' or 'Right'
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                if side == 'Left':
                    left_hand_coords = coords
                elif side == 'Right':
                    right_hand_coords = coords
        return np.concatenate([left_hand_coords.flatten(), right_hand_coords.flatten()])

    @torch.no_grad()
    def predict(self, frames: list, top_k: int = 3) -> dict:
        """
        frames: list of np.ndarray (BGR, uint8) — already sampled to NUM_FRAMES
        Returns {"predicted_class": str, "confidence": float, "top_k": [...], "probs": [...]}
        """
        if len(frames) != NUM_FRAMES:
            raise ValueError(f"ActionRecognizer expected exactly {NUM_FRAMES} frames, got {len(frames)}.")
            
        pil_frames = []
        landmarks_seq = []
        
        for f in frames:
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(frame_rgb))
            
            # Extract landmarks for the frame
            landmarks = self._extract_landmarks(frame_rgb)
            landmarks_seq.append(landmarks)
            
        # Preprocess images for VideoMAE
        inputs = self.processor(images=pil_frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        # Convert landmarks to tensor
        landmarks_tensor = torch.tensor(np.array(landmarks_seq), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Inference
        logits = self.model(pixel_values, landmarks_tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        
        top_i = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        return {
            "predicted_class": self.id2label[top_i[0]],
            "confidence": round(probs[top_i[0]], 4),
            "top_k": [(self.id2label[i], round(probs[i], 4)) for i in top_i],
            "probs": probs,
        }