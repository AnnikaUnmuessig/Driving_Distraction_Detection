#Data Preparation
#Balancing the dataset
#Removing long videos 
#Removing classes with too few samples or unclassified
import os
import cv2
import random
import csv


# Use absolute path relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "DMD_dataset")
#limit 'talking_to_passenger', 'safe_driving', and 'reach_side'
#not consider 'stand_still_waiting', 'reach_backseat', and 'unclassified'
CLASS_MAP = {
    "safe_driving": 0, "texting_right": 1, "phonecall_right": 2, 
    "texting_left": 3, "phonecall_left": 4, "radio": 5, 
    "drinking": 6, "reach_side": 7, "hair_and_makeup": 8, 
    "talking_to_passenger": 9, "change_gear": 10
} #11 classes
LIMIT_CAP = 120 #limit of samples per class


#csv files already created
#Desired format: video_path, label (0-10)
#70% train, 20% val, 10% test

def get_video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 999

all_entries = []
for class_name, label in CLASS_MAP.items():
    class_path = os.path.join(DATA_ROOT, class_name)
    if not os.path.exists(class_path): 
        continue
    
    videos = [v for v in os.listdir(class_path) if v.endswith(('.mp4', '.avi'))]
    
    #Get durations and sort
    video_stats = []
    for v in videos:
        dur = get_video_duration(os.path.join(class_path, v))
        video_stats.append((os.path.join(class_name, v), dur))
    
    #Sort by duration (shortest first) and cap if needed
    video_stats.sort(key=lambda x: x[1])
    selected = video_stats[:LIMIT_CAP] if len(video_stats) > LIMIT_CAP else video_stats
    
    for path, _ in selected:
        all_entries.append([os.path.join("DMD_dataset", path), label])

print(f"\nTotal entries collected: {len(all_entries)}")

#Shuffle and Split (70/20/10)
random.seed(42)
random.shuffle(all_entries)

total = len(all_entries)
train_idx = int(total * 0.7)
val_idx = int(total * 0.9)

splits = {
    'train.csv': all_entries[:train_idx],
    'val.csv': all_entries[train_idx:val_idx],
    'test.csv': all_entries[val_idx:]
}

# Ensure output directory exists
os.makedirs(DATA_ROOT, exist_ok=True)

for filename, rows in splits.items():
    filepath = os.path.join(DATA_ROOT, filename)
    with open(filepath, 'w', newline='') as f:
        for row in rows:
            f.write(f"{row[0]} {row[1]}\n")
    print(f"Wrote {len(rows)} rows to {filepath}")
