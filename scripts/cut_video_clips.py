"""cut_video_clips.py
===================
Cut a DMD body-camera MP4 into short clips using the corresponding
body-only OpenLABEL annotation JSON, and emit a TimeSformer-compatible
label file.

Usage
-----
    python scripts/cut_video_clips.py <folder>

Where <folder> contains pairs of files like:
    gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body.mp4
    gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction_body_only.json

Output (written inside <folder>/clips/):
    <clip_name>.mp4           – the trimmed clip
    labels.txt                – "<relative_clip_path> <class_id>" per line

Requirements
------------
    pip install opencv-python tqdm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Class map (matches TimeSformer convention: integer ids)
# ---------------------------------------------------------------------------
DMD_BODY_ACTION_MAP: dict[str, int] = {
    "safe_driving":         0,
    "texting_right":        1,
    "phonecall_right":      2,
    "texting_left":         3,
    "phonecall_left":       4,
    "radio":                5,
    "drinking":             6,
    "reach_side":           7,
    "hair_and_makeup":      8,
    "talking_to_passenger": 9,
    "reach_backseat":       10,
    "change_gear":          11,
    "stand_still_waiting":  12,
    "unclassified":         13,
}

# Pattern that matches the video stem and lets us derive the annotation stem:
#   gA_1_s1_2019-03-08T09;31;15+01;00_rgb_body
#   →  gA_1_s1_2019-03-08T09;31;15+01;00_rgb_ann_distraction_body_only
_VIDEO_STEM_RE = re.compile(r"^(.+)_rgb_body$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """Return (video, json) pairs found in *folder*."""
    pairs: list[tuple[Path, Path]] = []
    for mp4 in sorted(folder.glob("*.mp4")):
        m = _VIDEO_STEM_RE.match(mp4.stem)
        if not m:
            continue
        prefix = m.group(1)
        json_path = folder / f"{prefix}_rgb_ann_distraction_body_only.json"
        if json_path.exists():
            pairs.append((mp4, json_path))
        else:
            print(f"[WARN] No annotation JSON for {mp4.name} – skipping.")
    return pairs


def load_actions(json_path: Path) -> list[dict[str, Any]]:
    """Return a flat list of {type, frame_start, frame_end} dicts."""
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    actions_raw: dict[str, Any] = (
        payload.get("openlabel", {}).get("actions", {})
    )

    actions: list[dict[str, Any]] = []
    for action_id, action_data in actions_raw.items():
        action_type: str = str(action_data.get("type", "unclassified")).lower()
        for interval in action_data.get("frame_intervals", []):
            frame_start: int = int(interval["frame_start"])
            frame_end: int   = int(interval["frame_end"])
            actions.append(
                {
                    "id":          action_id,
                    "type":        action_type,
                    "frame_start": frame_start,
                    "frame_end":   frame_end,
                }
            )

    # Sort chronologically
    actions.sort(key=lambda a: (a["frame_start"], a["frame_end"]))
    return actions


def write_clip(
    cap: cv2.VideoCapture,
    out_path: Path,
    frame_start: int,
    frame_end: int,
    fps: float,
    width: int,
    height: int,
) -> bool:
    """Write frames [frame_start, frame_end] (inclusive) to *out_path*.

    Returns True on success, False if no frames were written.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frames_written = 0

    for _ in range(frame_end - frame_start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1

    out.release()

    if frames_written == 0:
        out_path.unlink(missing_ok=True)
        return False
    return True


def sanitize_label(label: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", label)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_pair(
    video_path: Path,
    json_path: Path,
    clips_dir: Path,
    label_lines: list[str],
    skip_unclassified: bool,
) -> None:
    """Cut *video_path* into clips according to *json_path*."""

    actions = load_actions(json_path)
    if not actions:
        print(f"[WARN] No actions found in {json_path.name} – skipping.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path.name} – skipping.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0  # sensible fallback

    print(
        f"\n→ {video_path.name}\n"
        f"   {total} frames  |  {fps:.2f} fps  |  {width}×{height}\n"
        f"   {len(actions)} action interval(s) in {json_path.name}"
    )

    # Unique output sub-folder per video so clip names never collide
    video_clips_dir = clips_dir / video_path.stem
    video_clips_dir.mkdir(parents=True, exist_ok=True)

    for idx, action in enumerate(
        tqdm(actions, desc="  cutting clips", unit="clip", leave=False)
    ):
        label_str  = action["type"]
        class_id   = DMD_BODY_ACTION_MAP.get(label_str, DMD_BODY_ACTION_MAP["unclassified"])

        if skip_unclassified and class_id == DMD_BODY_ACTION_MAP["unclassified"]:
            continue

        frame_start = max(0, action["frame_start"])
        frame_end   = min(total - 1, action["frame_end"])

        if frame_start > frame_end:
            continue

        safe_label = sanitize_label(label_str)
        clip_name  = f"{video_path.stem}_clip{idx:04d}_{safe_label}.mp4"
        out_path   = video_clips_dir / clip_name

        success = write_clip(cap, out_path, frame_start, frame_end, fps, width, height)

        if success:
            # Relative path from clips_dir root, using forward slashes (TimeSformer convention)
            rel = out_path.relative_to(clips_dir).as_posix()
            label_lines.append(f"{rel} {class_id}")

    cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Cut DMD body-camera videos into per-action clips using "
            "body-only OpenLABEL JSON annotations, and produce a "
            "TimeSformer-compatible label file."
        )
    )
    p.add_argument(
        "folder",
        type=Path,
        help="Folder containing the .mp4 and the _body_only.json files.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where clips/ sub-folder and labels.txt are written. "
            "Defaults to <folder>/clips."
        ),
    )
    p.add_argument(
        "--skip-unclassified",
        action="store_true",
        default=False,
        help="Do not write clips whose action type is 'unclassified'.",
    )
    return p


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    folder: Path = args.folder.resolve()

    if not folder.is_dir():
        sys.exit(f"[ERROR] '{folder}' is not a directory.")

    clips_dir: Path = (
        args.output_dir.resolve() if args.output_dir else folder / "clips"
    )
    clips_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(folder)
    if not pairs:
        sys.exit(
            "[ERROR] No matching (video, JSON) pairs found in the folder.\n"
            "        Expected files like:\n"
            "          *_rgb_body.mp4\n"
            "          *_rgb_ann_distraction_body_only.json"
        )

    label_lines: list[str] = []

    for video_path, json_path in pairs:
        process_pair(
            video_path,
            json_path,
            clips_dir,
            label_lines,
            skip_unclassified=args.skip_unclassified,
        )

    # Write labels file
    labels_path = clips_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(label_lines))
        if label_lines:
            fh.write("\n")

    print(
        f"\n✓ Done.\n"
        f"  Clips written to : {clips_dir}\n"
        f"  Label file       : {labels_path}\n"
        f"  Total clips      : {len(label_lines)}"
    )


if __name__ == "__main__":
    main()
