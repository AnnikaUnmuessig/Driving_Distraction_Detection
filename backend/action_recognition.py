import os
import sys
import torch
from pathlib import Path
from typing import Any, Callable
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import cv2, numpy as np
from PIL import Image
import torch.nn.functional as F
from annotate_video import annotate_video
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into environment

HF_TOKEN = os.getenv("HF_TOKEN")
from huggingface_hub import login


login(token=HF_TOKEN)

#Helper functions
NUM_FRAMES = 16
def load_video_frames(path, n=NUM_FRAMES):
    cap=cv2.VideoCapture(path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    idxs=np.linspace(0,max(total-1,0),n,dtype=int); frames=[]
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i))
        ret,fr=cap.read()
        if ret: frames.append(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames: frames=[Image.new('RGB',(224,224))]*n
    while len(frames)<n: frames.append(frames[-1])
    return frames[:n]
print('load_video_frames() ready.')


@torch.no_grad()
def predict_video(path, top_k=3):
    frames=load_video_frames(path)
    inputs=processor(images=frames,return_tensors='pt')
    inputs={k:v.to(DEVICE) for k,v in inputs.items()}
    probs=F.softmax(model(**inputs).logits,dim=-1).squeeze(0).cpu().tolist()
    top_i=sorted(range(len(probs)),key=lambda i:probs[i],reverse=True)[:top_k]
    return {'predicted_class':ID2LABEL[top_i[0]],'confidence':round(probs[top_i[0]],4),
            'top_k':[(ID2LABEL[i],round(probs[i],4)) for i in top_i]}
print('predict_video() ready.')


def annotate_video_videomae(INPUT_VIDEO='', OUTPUT_VIDEO='', PREDICTION_INTERVAL_SEC=1.0, FRAMES_PER_SEGMENT=16):
    if not INPUT_VIDEO:
        print('⚠ Set INPUT_VIDEO.')
        return None
    saved = annotate_video(
        model_dir=MODEL_DIR,
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        interval_sec=PREDICTION_INTERVAL_SEC,
        num_frames=FRAMES_PER_SEGMENT,
        model_class='videomae'
    )
    print(f'Done: {saved}')
    return saved


MODEL_DIR = Path(__file__).resolve().parent.parent / 'models' / 'video_mae'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = Path(__file__).resolve().parent.parent

if not MODEL_DIR.exists():
    raise FileNotFoundError(f'Local model directory not found: {MODEL_DIR}')

BASE = 'MCG-NJU/videomae-base-finetuned-kinetics'
processor = VideoMAEImageProcessor.from_pretrained(BASE, token=HF_TOKEN)
print(f'Loading weights from: {MODEL_DIR}')
model = VideoMAEForVideoClassification.from_pretrained(str(MODEL_DIR), token=HF_TOKEN)
model.eval().to(DEVICE)
ID2LABEL = model.config.id2label
print(f'\nModel loaded — {model.config.num_labels} classes:')
[print(f'  [{i:2d}] {ID2LABEL[i]}') for i in sorted(ID2LABEL.keys())]

# Prediction
VIDEO_PATH = ROOT_DIR / 'test_data' / 'test_video.mp4'
if not VIDEO_PATH.exists():
    print(f'⚠ {VIDEO_PATH} not found. Set VIDEO_PATH to a valid video file to run prediction.')
else:
    r = predict_video(str(VIDEO_PATH))
    print(f'\n{VIDEO_PATH.name}')
    print(f'  → {r["predicted_class"]}  ({r["confidence"]*100:.1f}%)')
    for cls, score in r['top_k']:
        print(f'     {cls:25s} {score*100:5.1f}%  {"█"*int(score*30)}')

OUTPUT_VIDEO = ''
PREDICTION_INTERVAL_SEC = 1.0
FRAMES_PER_SEGMENT = 16
# Annotate only when an input video is provided.
if VIDEO_PATH.exists():
    saved = annotate_video_videomae(
        INPUT_VIDEO=str(VIDEO_PATH),
        OUTPUT_VIDEO=OUTPUT_VIDEO,
        PREDICTION_INTERVAL_SEC=PREDICTION_INTERVAL_SEC,
        FRAMES_PER_SEGMENT=FRAMES_PER_SEGMENT,
    )
else:
    print('⚠ No valid VIDEO_PATH set. Skipping annotation.')
