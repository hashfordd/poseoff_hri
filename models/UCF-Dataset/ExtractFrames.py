import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from glob import glob
import numpy as np
import io
import torch
from utils import flowpose_lk, flowpose_lk, get_poses
from ultralytics import YOLO

def process_single_video(args):
    video_path, dataset_dir = args

    with open(video_path, 'rb') as f:
        video =  f.read()

    cap = cv2.VideoCapture(video) 

    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)

    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    return frames, dataset_dir

def apply_poseOFF(video_sequence, pose_model, window_size, threshold, dilation):
    
    print(f"Total frames: {len(video_sequence)}")

    flow_sequence = []
                
    for i in range(len(video_sequence) - 1):        
        frame1 = video_sequence[i]
        if len(frame1.shape) == 2 or frame1.shape[2] == 1:
            frame1_grey = frame1.squeeze()
        else:
            frame1_grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        frame2 = video_sequence[i+1]
        if len(frame2.shape) == 2 or frame2.shape[2] == 1:
            frame2_grey = frame2.squeeze()
        else:
            frame2_grey = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if frame1 is None or frame2  is None:
            continue
                
        poses = get_poses(frame1_grey, pose_model)

        flow_poses, p0, p1 = flowpose_lk(frame1_grey, frame2_grey, poses,  window_size, threshold, dilation)
          
        if not isinstance(flow_poses, torch.Tensor):
            flow_poses = torch.tensor(flow_poses, device=poses.device)
            flow_sequence.append(flow_poses)
            
        if len(flow_sequence) > 0:
            video_tensor = torch.stack(flow_sequence, dim=0) # Shape: (T, C*W, V, M)
            # T is the number of frames in video_sequence,
            # C*W is the Cordinates of the Flow chennels * Window_size
            # V Is the numbero of Joints
            # M is number of people in frame 
            
            # Convert Tensor to numpy then save
            video_tensor = video_tensor.detach().cpu().numpy()
            return video_tensor

def extract_dataset_frames(dataset_dir, output_dir, max_workers):
    print("Searching for UCF101 .avi files...")
    video_paths = glob(os.path.join(dataset_dir, "**", "*.avi"), recursive=True)
    print(f"Found {len(video_paths)} videos.")
    
    tasks = [(v, dataset_dir) for v in video_paths]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, result in enumerate(executor.map(process_single_video, tasks)):
            dataset_frames, filenames = result

    for idx, video_frames in enumerate(dataset_frames):
        video_numpy = apply_poseOFF(video_frames, pose_model, window_size=5, threshold=0.2, dilation=1)

        if video_numpy is not None:
            rel_path = os.path.relpath(filenames, start=dataset_dir)
            category_dir = os.path.dirname(rel_path)
            video_name = os.path.splitext(os.path.basename(filenames))[0]

            target_dir = os.path.join(output_dir, category_dir)
            os.makedirs(target_dir, exist_ok=True)

            output_file_path = os.path.join(target_dir, f"{video_name}.npy")
            np.save(output_file_path, video_numpy)
            print(output_file_path)

    
    print("Frame extraction complete! Processing PoseOFF")
    
if __name__ == "__main__":
    pose_model = YOLO("yolo26m-pose.pt")

    extract_dataset_frames("./ucf101/UCF-101/UCF-101", './ucf-numpy', 16)
