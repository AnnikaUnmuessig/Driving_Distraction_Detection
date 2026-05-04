"""
analyze_dataset.py
==================
Counts total frames per class in endoard/distraction_detection_dataset.

Two modes:
  --mode local   : reads videos from a local folder (fast, requires download)
  --mode stream  : streams videos directly from HuggingFace (no download needed, slower)

Usage:
    # If dataset already downloaded:
    python analyze_dataset.py --mode local --dataset_path ./distraction_dataset

    # If dataset NOT downloaded yet (streams from HF):
    python analyze_dataset.py --mode stream

    # Save results to CSV:
    python analyze_dataset.py --mode local --dataset_path ./distraction_dataset --csv results.csv
    python analyze_dataset.py --mode stream --csv frame_stats.csv
"""

import argparse
import os
import sys
import csv
import tempfile
from collections import defaultdict

import cv2

# ── Classes (consistent with Finetuning.py) ────────────────────────────────────

ACTIVE_CLASSES = [
    "safe_driving",
    "texting_right",
    "phonecall_right",
    "texting_left",
    "phonecall_left",
    "radio",
    "drinking",
    "reach_side",
    "hair_and_makeup",
    "talking_to_passenger",
    "change_gear",
]

EXCLUDED_CLASSES = ["reach_backseat", "stand_still_waiting", "unclassified"]

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


# ── Frame counting ─────────────────────────────────────────────────────────────

def count_frames_cv2(video_path: str) -> int:
    """Count frames in a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    # CAP_PROP_FRAME_COUNT is fast (reads container metadata, no decoding)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Some containers return 0 or -1; fall back to manual count only then
    if count <= 0:
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()
    return count


# ── Local mode ─────────────────────────────────────────────────────────────────

def analyze_local(dataset_path: str, include_excluded: bool = False) -> dict:
    """
    Scan dataset_path/<class_name>/ directories and count frames per class.
    Returns {class_name: {"clips": int, "frames": int, "fps_list": list}}
    """
    all_classes = ACTIVE_CLASSES + (EXCLUDED_CLASSES if include_excluded else [])
    stats = {}

    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        if class_name not in all_classes:
            print(f"  [SKIP] {class_name} (not in class list)")
            continue

        videos = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
        total_frames = 0
        fps_list = []
        for i, fname in enumerate(videos, 1):
            video_path = os.path.join(class_dir, fname)
            n = count_frames_cv2(video_path)
            total_frames += n

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps_list.append(fps)

            print(f"\r  {class_name}: {i}/{len(videos)} clips — {total_frames} frames so far", end="", flush=True)

        print()  # newline after progress
        stats[class_name] = {
            "clips": len(videos),
            "frames": total_frames,
            "fps_list": fps_list,
        }

    return stats


# ── Streaming mode ─────────────────────────────────────────────────────────────

def analyze_stream(hf_repo: str = "endoard/distraction_detection_dataset",
                   include_excluded: bool = False) -> dict:
    """
    Stream videos directly from HuggingFace Hub, count frames per class.
    Downloads each video to a shared temp dir, counts frames, then deletes it.
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        from tqdm import tqdm
    except ImportError:
        sys.exit("Run: pip install huggingface_hub tqdm")

    all_classes = ACTIVE_CLASSES + (EXCLUDED_CLASSES if include_excluded else [])

    print(f"Listing files in {hf_repo}...")
    all_files = list(list_repo_files(hf_repo, repo_type="dataset"))
    video_files = [
        f for f in all_files
        if f.lower().endswith(VIDEO_EXTENSIONS)
        and f.replace("\\", "/").split("/")[0] in all_classes
    ]
    print(f"Found {len(video_files)} video files to process.\n")

    stats = defaultdict(lambda: {"clips": 0, "frames": 0, "fps_list": []})

    # Single shared temp dir — reuse across all downloads
    tmp_dir = tempfile.mkdtemp(prefix="hf_analyze_")

    with tqdm(video_files, unit="video", dynamic_ncols=True) as pbar:
        for fpath in pbar:
            parts = fpath.replace("\\", "/").split("/")
            class_name = parts[0]

            pbar.set_description(f"{class_name:<28}")

            try:
                local_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=fpath,
                    repo_type="dataset",
                    local_dir=tmp_dir,
                )
                n = count_frames_cv2(local_path)
                cap = cv2.VideoCapture(local_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                try:
                    os.remove(local_path)
                except OSError:
                    pass

                stats[class_name]["clips"] += 1
                stats[class_name]["frames"] += n
                stats[class_name]["fps_list"].append(fps)

                pbar.set_postfix({
                    "class": class_name,
                    "clips": stats[class_name]["clips"],
                    "frames": stats[class_name]["frames"],
                })
            except Exception as e:
                tqdm.write(f"  [ERROR] {fpath}: {e}")

    # Cleanup temp dir
    try:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return dict(stats)


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_report(stats: dict, csv_path: str = None):
    """Print a formatted table and optionally save to CSV."""
    if not stats:
        print("No data collected.")
        return

    # Sort by total frames descending
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["frames"], reverse=True)

    total_frames_all = sum(v["frames"] for v in stats.values())
    total_clips_all  = sum(v["clips"]  for v in stats.values())

    print("\n" + "=" * 80)
    print(f"{'CLASS':<28} {'CLIPS':>8} {'TOTAL FRAMES':>14} {'AVG FRAMES/CLIP':>17} {'% FRAMES':>10}")
    print("-" * 80)

    rows = []
    for class_name, s in sorted_stats:
        clips  = s["clips"]
        frames = s["frames"]
        avg    = frames / clips if clips > 0 else 0
        pct    = 100 * frames / total_frames_all if total_frames_all > 0 else 0
        fps_avg = sum(s["fps_list"]) / len(s["fps_list"]) if s["fps_list"] else 0
        print(f"  {class_name:<26} {clips:>8} {frames:>14,} {avg:>17.1f} {pct:>9.1f}%")
        rows.append({
            "class": class_name,
            "clips": clips,
            "total_frames": frames,
            "avg_frames_per_clip": round(avg, 1),
            "pct_frames": round(pct, 2),
            "avg_fps": round(fps_avg, 2),
        })

    print("-" * 80)
    print(f"  {'TOTAL (active classes)':<26} {total_clips_all:>8} {total_frames_all:>14,}")
    print("=" * 80)

    # Suggest a cap
    min_frames = min(v["frames"] for v in stats.values())
    print(f"\n💡 Balancing insight:")
    print(f"   Smallest class frames : {min_frames:,}")
    print(f"   If you cap ALL classes to {min_frames:,} frames → {min_frames * len(stats):,} total frames")

    # Suggest clip caps
    print(f"\n   Suggested LIMIT_CAP per class (to reach ~{min_frames:,} frames each):")
    for class_name, s in sorted_stats:
        avg = s["frames"] / s["clips"] if s["clips"] > 0 else 1
        suggested_clips = int(min_frames / avg) if avg > 0 else s["clips"]
        suggested_clips = min(suggested_clips, s["clips"])
        print(f"     {class_name:<28}: {suggested_clips:>4} clips  (avg {avg:.0f} frames/clip)")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ Results saved to: {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Count total frames per class in the distraction detection dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["local", "stream"],
        default="local",
        help="'local' reads from a downloaded folder; 'stream' downloads one-at-a-time from HF.",
    )
    parser.add_argument(
        "--dataset_path",
        default="./distraction_dataset",
        help="Path to the local dataset root (used in --mode local).",
    )
    parser.add_argument(
        "--include_excluded",
        action="store_true",
        help="Also analyze excluded classes (reach_backseat, stand_still_waiting, unclassified).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional: path to save results as CSV (e.g., frame_stats.csv).",
    )
    args = parser.parse_args()

    print(f"Mode: {args.mode}")

    if args.mode == "local":
        dataset_path = os.path.abspath(args.dataset_path)
        if not os.path.isdir(dataset_path):
            sys.exit(f"ERROR: Dataset folder not found: {dataset_path}\n"
                     "Run download_assets.py first, or use --mode stream.")
        print(f"Dataset path: {dataset_path}\n")
        stats = analyze_local(dataset_path, include_excluded=args.include_excluded)
    else:
        stats = analyze_stream(include_excluded=args.include_excluded)

    print_report(stats, csv_path=args.csv)


if __name__ == "__main__":
    main()
