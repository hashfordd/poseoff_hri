import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from glob import glob

def process_single_video(args):
    video_path, output_root = args
    
    # Maintain folder structure (ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi)
    rel_path = os.path.relpath(video_path, start=output_root)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    category_dir = os.path.dirname(rel_path)
    
    target_dir = os.path.join(output_root, "frames", category_dir, video_name)
    os.makedirs(target_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_filename = os.path.join(target_dir, f"frame_{frame_idx:05d}.jpg")
        if not os.path.exists(frame_filename):
            cv2.imwrite(frame_filename, frame)
            
        frame_idx += 1
        
    cap.release()

def extract_dataset_frames(dataset_dir, max_workers=8):
    print("Searching for UCF101 .avi files...")
    video_paths = glob(os.path.join(dataset_dir, "**", "*.avi"), recursive=True)
    print(f"Found {len(video_paths)} videos. Beginning extraction...")
    
    tasks = [(v, dataset_dir) for v in video_paths]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process_single_video, tasks))
        
    print("Frame extraction complete!")

if __name__ == "__main__":
    extract_dataset_frames("./ucf101/UCF-101", max_workers=8)
