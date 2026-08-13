import os 
import glob
import torch
from tqdm import tqdm
import numpy as np
import cv2
from utils import flowpose_lk, flowpose_lk, get_poses
from ultralytics import YOLO

def apply_poseOFF(frame_dir, output_dir, pose_model, window_size, threshold, dilation):
    os.makedirs(output_dir, exist_ok= True)

    video_dirs = sorted([d for d in glob.glob(os.path.join(frame_dir, "*", "*")) if os.path.isdir(d)])

    for dir in tqdm(video_dirs):
        # Get category and video name from path structure
        parts = os.path.normpath(dir).split(os.sep)
        category, video_name = parts[-2], parts[-1]
        
        # Output path
        out_dir = os.path.join(output_dir, category)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{video_name}.npy")
            
        frame_paths = sorted(glob.glob(os.path.join(dir, "*.jpg"))) 
        print(f" Total frames: {len(frame_paths)}")
        
        if not frame_paths:
            continue
            
        video_sequence = []

   
        for i in range(len(frame_paths) - 1):        
            frame1 = cv2.imread(frame_paths[i])
            frame1_grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            frame2 = cv2.imread(frame_paths[i + 1])
            frame2_grey = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            if frame1 is None or frame2  is None:
                continue
                
            poses = get_poses(frame1_grey, pose_model)

            flow_poses, p0, p1 = flowpose_lk(frame1_grey, frame2_grey, poses,  window_size, threshold, dilation)
          
            if not isinstance(flow_poses, torch.Tensor):
                flow_poses = torch.tensor(flow_poses, device=poses.device)

            video_sequence.append(flow_poses)
            
        if len(video_sequence) > 0:
            video_tensor = torch.stack(video_sequence, dim=0) # Shape: (T, C*W, V, M)
            # T is the number of frames in video_sequence,
            # C*W is the Cordinates of the Flow chennels * Window_size
            # V Is the numbero of Joints
            # M is number of people in frame 
            
            # Convert Tensor to numpy then save
            video_tensor = video_tensor.detach().cpu().numpy()
            np.save(out_file, video_tensor)

if __name__ == "__main__":
    pose_model = YOLO("yolo26m-pose.pt")
    apply_poseOFF("./frames/UCF-101", "./completed_flow_poses", pose_model, window_size=5, threshold=0.2, dilation=1)
