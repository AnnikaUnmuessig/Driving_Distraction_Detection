"""
annotate_video.py
=================
Run sliding-window inference on a video and produce an annotated copy
with the predicted distraction class overlaid on every frame.

Supports both TimeSformer and VideoMAE checkpoints via --model_class.

How it works
------------
The video is split into non-overlapping segments of `PREDICTION_INTERVAL_SEC`
seconds. For each segment, NUM_FRAMES frames are sampled uniformly and passed
to the fine-tuned model. The resulting label and confidence are drawn on every
frame of that segment. The annotated frames are written to `OUTPUT_PATH`.

Usage (command line)
--------------------
    python annotate_video.py \\
        --model_class timesformer \\
        --model_dir   ./timesformer_outputs/best_model \\
        --input       ./my_video.mp4 \\
        --output      ./annotated.mp4 \\
        --interval    1.0

    python annotate_video.py \\
        --model_class videomae \\
        --model_dir   ./videomae_outputs/best_model \\
        --input       ./my_video.mp4

Usage (from a notebook cell)
-----------------------------
    from annotate_video import annotate_video
    annotate_video(model_class='videomae', model_dir=MODEL_DIR,
                   input_path=INPUT_VIDEO, interval_sec=1.0)
"""

import os
import argparse
import cv2
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)
import torch.nn.functional as F
from PIL import Image

# Default parameters (overridden by CLI args) 
DEFAULT_MODEL_CLASS = 'timesformer'   # 'timesformer' or 'videomae'
DEFAULT_MODEL_DIR   = './timesformer_outputs/best_model'
DEFAULT_INPUT       = ''              # path to input video
DEFAULT_OUTPUT      = ''              # path for annotated output ('' → auto-named)
DEFAULT_INTERVAL    = 1.0             # seconds between predictions
DEFAULT_NUM_FRAMES  = 16              # frames sampled per segment (must match training)

_FALLBACK = {
    'timesformer': 'facebook/timesformer-hr-finetuned-k400',
    'videomae':    'MCG-NJU/videomae-base-finetuned-kinetics',
}

# Overlay style 
BOX_ALPHA       = 0.55    # transparency of the label background box
BOX_COLOR_BGR   = (20, 20, 20)
TEXT_COLOR_BGR  = (255, 255, 255)
CONF_COLOR_BGR  = (100, 230, 100)
FONT_SCALE      = 0.75
FONT_THICKNESS  = 2
FONT            = cv2.FONT_HERSHEY_DUPLEX


def load_model(model_dir: str, model_class: str = 'timesformer'):
    """Load processor and fine-tuned model (TimeSformer or VideoMAE).

    Falls back to HF Hub for the processor if preprocessor_config.json is absent.
    """
    model_class = model_class.lower()
    if model_class not in _FALLBACK:
        raise ValueError(f"model_class must be 'timesformer' or 'videomae', got '{model_class}'")

    fallback_id = _FALLBACK[model_class]
    proc_cfg    = os.path.join(model_dir, 'preprocessor_config.json')

    # Fix filename saved without trailing 's'
    _wrong = os.path.join(model_dir, 'model.safetensor')
    _right = os.path.join(model_dir, 'model.safetensors')
    if os.path.isfile(_wrong) and not os.path.isfile(_right):
        import shutil; shutil.copy2(_wrong, _right)
        print(f'[INFO] Renamed model.safetensor → model.safetensors')

    if model_class == 'videomae':
        ProcessorCls = VideoMAEImageProcessor
        ModelCls     = VideoMAEForVideoClassification
    else:
        ProcessorCls = AutoImageProcessor
        ModelCls     = TimesformerForVideoClassification

    if os.path.isfile(proc_cfg):
        processor = ProcessorCls.from_pretrained(model_dir)
        print(f'[INFO] Processor loaded from local checkpoint.')
    else:
        processor = ProcessorCls.from_pretrained(fallback_id)
        print(f'[INFO] preprocessor_config.json not found — processor loaded from HF Hub ({fallback_id}).')

    model  = ModelCls.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'[INFO] {model_class} loaded ({model.config.num_labels} classes) on {device}.')
    return processor, model, device


def sample_frames(pil_frames: list, num_frames: int) -> list:
    """Uniformly sample `num_frames` frames from a list of PIL Images."""
    n = len(pil_frames)
    if n == 0:
        return [Image.new('RGB', (448, 448))] * num_frames
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    sampled = [pil_frames[i] for i in indices]
    # Pad if needed
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])
    return sampled[:num_frames]


@torch.no_grad()
def infer_segment(pil_frames: list, processor, model, device, num_frames: int):
    """Run inference on a list of PIL frames. Returns (label, confidence, top3)."""
    sampled = sample_frames(pil_frames, num_frames)
    inputs  = processor(images=sampled, return_tensors='pt')
    inputs  = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits                   # (1, num_classes)
    probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    id2label    = model.config.id2label
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    best_id     = top_indices[0]
    top3        = [(id2label[i], round(probs[i], 3)) for i in top_indices[:3]]
    return id2label[best_id], round(probs[best_id], 3), top3


def draw_overlay(frame_bgr: np.ndarray, label: str, confidence: float, top3: list) -> np.ndarray:
    """Draw a semi-transparent label box on a BGR frame (in-place copy)."""
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    #Background box 
    box_h = 90
    cv2.rectangle(overlay, (0, h - box_h), (w, h), BOX_COLOR_BGR, -1)
    frame_out = cv2.addWeighted(overlay, BOX_ALPHA, frame_bgr, 1 - BOX_ALPHA, 0)

    # Main label 
    label_text = label.replace('_', ' ').title()
    cv2.putText(frame_out, label_text,
                (12, h - box_h + 30), FONT, FONT_SCALE,
                TEXT_COLOR_BGR, FONT_THICKNESS, cv2.LINE_AA)

    #Model Confidence
    conf_text = f'{confidence * 100:.1f}%'
    cv2.putText(frame_out, conf_text,
                (12, h - box_h + 58), FONT, 0.6,
                CONF_COLOR_BGR, 1, cv2.LINE_AA)

    #Top-3 mini bar (right side) 
    bar_x = w - 240
    for rank, (cls, score) in enumerate(top3):
        bar_y    = h - box_h + 18 + rank * 24
        bar_len  = int(score * 180)
        bar_col  = CONF_COLOR_BGR if rank == 0 else (180, 180, 180)
        cv2.rectangle(frame_out, (bar_x, bar_y - 10), (bar_x + bar_len, bar_y + 4), bar_col, -1)
        short_cls = cls.replace('_', ' ')[:18]
        cv2.putText(frame_out, f'{short_cls} {score*100:.0f}%',
                    (bar_x, bar_y + 3), FONT, 0.38,
                    TEXT_COLOR_BGR, 1, cv2.LINE_AA)

    return frame_out


def annotate_video(
    model_dir: str, #Path to the best_model/ directory
    input_path: str, #Path to the source video file
    output_path: str = '', #Where to save the annotated video. Auto-named if empty.
    interval_sec: float = 1.0, #Seconds between prediction updates (default: 1.0).
    num_frames: int = 16, #Frames sampled per segment for inference (default: 16).
    model_class: str = 'timesformer', #'timesformer' or 'videomae' (default: 'timesformer'
):
    """Annotate a video with sliding-window distraction predictions."""

    
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f'{base}_annotated{ext}'

    print(f'[INFO] Input  : {input_path}')
    print(f'[INFO] Output : {output_path}')
    print(f'[INFO] Model  : {model_class} | Interval: {interval_sec}s | Frames/segment: {num_frames}')

    #Load model 
    processor, model, device = load_model(model_dir, model_class)

    #Open video 
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {input_path}')

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seg_frames   = max(1, int(round(fps * interval_sec)))   # frames per segment
    total_segs   = max(1, (total_frames + seg_frames - 1) // seg_frames)

    print(f'[INFO] FPS: {fps:.1f} | Total frames: {total_frames} | '
          f'Frames/seg: {seg_frames} | Total segments: {total_segs}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process segment by segment 
    seg_idx      = 0
    current_label = 'Initializing...'
    current_conf  = 0.0
    current_top3  = []

    while True:
        raw_frames = []
        pil_frames = []

        # Read one segment worth of frames
        for _ in range(seg_frames):
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            pil_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if not raw_frames:
            break

        seg_idx += 1
        print(f'  Segment {seg_idx:4d}/{total_segs}  ({len(raw_frames)} frames) ', end='', flush=True)

        # Inference on this segment
        current_label, current_conf, current_top3 = infer_segment(
            pil_frames, processor, model, device, num_frames
        )
        print(f'→ {current_label} ({current_conf*100:.1f}%)')

        # Write annotated frames
        for frame in raw_frames:
            annotated = draw_overlay(frame, current_label, current_conf, current_top3)
            out.write(annotated)

    cap.release()
    out.release()
    print(f'\n[DONE] Annotated video saved to: {output_path}')
    return output_path


#CLI entry-point

def parse_args():
    p = argparse.ArgumentParser(description='Annotate a video with distraction predictions.')
    p.add_argument('--model_class', default=DEFAULT_MODEL_CLASS,
                   choices=['timesformer', 'videomae'],
                   help='Model architecture (default: %(default)s).')
    p.add_argument('--model_dir', default=DEFAULT_MODEL_DIR,
                   help='Path to best_model/ directory (default: %(default)s).')
    p.add_argument('--input',     required=True,
                   help='Path to the input video file.')
    p.add_argument('--output',    default='',
                   help='Path for the annotated output video (default: auto-named next to input).')
    p.add_argument('--interval',  type=float, default=DEFAULT_INTERVAL,
                   help='Seconds between prediction updates (default: %(default)s).')
    p.add_argument('--num_frames', type=int, default=DEFAULT_NUM_FRAMES,
                   help='Frames sampled per segment (default: %(default)s).')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    annotate_video(
        model_dir    = args.model_dir,
        input_path   = args.input,
        output_path  = args.output,
        interval_sec = args.interval,
        num_frames   = args.num_frames,
        model_class  = args.model_class,
    )
