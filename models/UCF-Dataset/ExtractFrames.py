import os
import cv2
from collections import deque
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
    return frames, video_path

def apply_poseOFF(video_sequence, pose_model, window_size, threshold, dilation):

    print(f"Total frames: {len(video_sequence)}")

    flow_sequence = []

    for i in range(len(video_sequence) - 1):
        frame1 = video_sequence[i]
        frame2 = video_sequence[i+1]

        # Guard before the colour conversion, not after it: cvtColor(None) raises
        # rather than falling through to the check.
        if frame1 is None or frame2 is None:
            continue

        if len(frame1.shape) == 2 or frame1.shape[2] == 1:
            frame1_grey = frame1.squeeze()
        else:
            frame1_grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if len(frame2.shape) == 2 or frame2.shape[2] == 1:
            frame2_grey = frame2.squeeze()
        else:
            frame2_grey = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Pose detection runs on the colour frame, matching pose.py. YOLO pose
        # expects 3 channel BGR; the greyscale frames exist only for flowpose_lk.
        poses = get_poses(frame1, pose_model, threshold=threshold)

        flow_poses, p0, p1 = flowpose_lk(frame1_grey, frame2_grey, poses,  window_size, threshold, dilation)

        # The append must not be nested in the isinstance check. It only ever ran
        # because flowpose_lk happens to return numpy; if that changed the
        # sequence would silently stay empty.
        if not isinstance(flow_poses, torch.Tensor):
            flow_poses = torch.tensor(flow_poses, device=poses.device)
        flow_sequence.append(flow_poses)

    if not flow_sequence:
        return None

    # Stack once, after the whole clip has been walked. Shape: (T, C*W, V, M)
    # T is len(video_sequence) - 1, one flow field per consecutive frame pair
    # C*W is the flow coordinate channels * window_size**2
    # V is the number of joints
    # M is the number of people in frame
    video_tensor = torch.stack(flow_sequence, dim=0)

    # Convert Tensor to numpy then save
    return video_tensor.detach().cpu().numpy()

def extract_dataset_frames(dataset_dir, output_dir, max_workers, pose_model,
                           window_size=5, threshold=0.2, dilation=1):
    print("Searching for UCF101 .avi files...")
    video_paths = sorted(glob(os.path.join(dataset_dir, "**", "*.avi"), recursive=True))
    print(f"Found {len(video_paths)} videos.")

    # Decoding runs in the worker threads, pose inference stays on the main thread
    # because the Ultralytics model is not safe to call concurrently.
    #
    # Keep only a bounded number of decoded videos in flight. executor.map would
    # submit all of them at once and buffer every decoded video in memory, which
    # is roughly 0.6 TB across UCF-101.
    max_in_flight = max(max_workers * 2, 1)

    saved = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        remaining = iter(video_paths)
        pending = deque()

        def submit_next():
            try:
                video_path = next(remaining)
            except StopIteration:
                return False
            pending.append(executor.submit(process_single_video, (video_path, dataset_dir)))
            return True

        for _ in range(max_in_flight):
            if not submit_next():
                break

        while pending:
            frames, video_path = pending.popleft().result()
            submit_next()

            if not frames:
                print(f"No frames decoded, skipping: {video_path}")
                failed += 1
                continue

            # Each video is handed over whole. Passing individual frames here is
            # what made apply_poseOFF index pixel rows as if they were frames.
            video_numpy = apply_poseOFF(frames, pose_model, window_size=window_size,
                                        threshold=threshold, dilation=dilation)

            if video_numpy is None:
                skipped += 1
                continue

            rel_path = os.path.relpath(video_path, start=dataset_dir)
            category_dir = os.path.dirname(rel_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            target_dir = os.path.join(output_dir, category_dir)
            os.makedirs(target_dir, exist_ok=True)

            output_file_path = os.path.join(target_dir, f"{video_name}.npy")
            np.save(output_file_path, video_numpy)
            saved += 1

    print(f"Frame extraction complete. saved={saved} skipped={skipped} failed={failed} "
          f"of {len(video_paths)} videos.")
    return saved

if __name__ == "__main__":
    pose_model = YOLO("yolo26m-pose.pt")

    extract_dataset_frames("./ucf101/UCF-101/UCF-101", './ucf-numpy', 16, pose_model)
