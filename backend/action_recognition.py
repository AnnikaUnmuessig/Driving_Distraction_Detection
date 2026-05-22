# action_recognition.py
import torch, cv2, numpy as np
from pathlib import Path
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch.nn.functional as F
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

NUM_FRAMES = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "video_mae"

class ActionRecognizer:
    def __init__(self):
        from huggingface_hub import login
        login(token=HF_TOKEN)
        BASE = "MCG-NJU/videomae-base-finetuned-kinetics"
        self.processor = VideoMAEImageProcessor.from_pretrained(BASE, token=HF_TOKEN)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            str(MODEL_DIR), token=HF_TOKEN
        ).eval().to(DEVICE)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, frames: list, top_k: int = 3) -> dict:
        """
        frames: list of np.ndarray (BGR, uint8) — already sampled to NUM_FRAMES
        Returns {"predicted_class": str, "confidence": float, "top_k": [...]}
        """
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]
        inputs = self.processor(images=pil_frames, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        probs = F.softmax(self.model(**inputs).logits, dim=-1).squeeze(0).cpu().tolist()
        top_i = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:top_k]
        return {
            "predicted_class": self.id2label[top_i[0]],
            "confidence": round(probs[top_i[0]], 4),
            "top_k": [(self.id2label[i], round(probs[i], 4)) for i in top_i],
        }