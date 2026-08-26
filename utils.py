#!/usr/bin/env python3

from xml.parsers.expat import model

from einops import rearrange
from ultralytics import YOLO
import torch
import numpy as np
import cv2


def get_norm_flows(img1, img2, alpha=1):
    # Calculate spatial gradients
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=5)

    # Temporal gradient
    It = img2.astype(float) - img1.astype(float)

    # Normal flow vectors
    # (must add small factor in demoninator to avoid div by zero error)
    norm_flow = -It / (alpha * (np.sqrt(Ix**2 + Iy**2) + 1e-6))

    return norm_flow


def resolve_device(device=None):
    """Pick an inference device, preferring an explicit choice.

    Ultralytics raises outright on device='cuda' when no NVIDIA GPU is present,
    so hardcoding it made the extraction script and the live demo unrunnable on
    CPU-only machines and Apple Silicon.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_poses(frame, pose_model, threshold=0.2, device=None):
    results = pose_model(frame, device=resolve_device(device), verbose=False)
    result = results[0]
    # data output shape: ((x,y,conf.), keypoints, bodies)
    poses = torch.zeros(3, 17, 2)
    for m, person in enumerate(result.keypoints):
        if m >= 2:
            break
        try:
            assert person.xyn.shape[1] == 17
        except AssertionError:
            continue
        # YOLO pose output is in the interval [0-1]
        poses[0, :, m] = person.xyn[0, :, 0]
        poses[1, :, m] = person.xyn[0, :, 1]
        poses[2, :, m] = person.conf[0]

        # set x and y to zero if confidence is zero
        poses[0][poses[2] < threshold] = 0
        poses[1][poses[2] < threshold] = 0

    poses = rearrange(poses, 'C V M -> (M V) C')
    return poses

def draw_bones(frame, pose, person_num=None):
    frame = frame.copy()
    H,W,C = frame.shape
    pose_local = pose.detach().clone()
    pose_local[:, 0] = pose_local[:, 0] * (W-1)
    pose_local[:, 1] = pose_local[:, 1] * (H-1)
    pose_local = rearrange(pose_local, '(M V) C -> M V C', M=2, V=17)
    joint_connections = [
        [0,1], [0,2], [1,3], [2,4],
        [5,6], [5,7], [7,9], [6,8], [8,10],
        [5,11], [11,13], [13,15],
        [6,12], [12,14], [14,16]
    ]
    # Check if alpha channel exists in frame
    color = (255,0,0) if frame.shape[-1] == 3 else (255,0,0,255)
    # Get individual person's specific pose (if person_num specified)
    if person_num in [0,1]:
        pose_local = pose_local[person_num].reshape((1, 17, 2))

    for person in pose_local:
        for joint_connection in joint_connections:
            p1, p2 = joint_connection
            if person[p1, 0] <= 1.0 or person[p2, 0] <= 1.0:
                continue
            cv2.line(frame,
                    (int(person[p1,0]), int(person[p1,1])),
                    (int(person[p2,0]), int(person[p2,1])),
                    color, 3
                    )
    return frame


def draw_skel(frame, pose, person_num=None, skip_points=[], debug=False):  # Poses shape: (M V) C
    frame = frame.copy()
    H,W,C = frame.shape
    pose_local = pose.detach().clone()
    pose_local[:, 0] = pose_local[:, 0] * (W-1)
    pose_local[:, 1] = pose_local[:, 1] * (H-1)
    pose_local = rearrange(pose_local, '(M V) C -> M V C', M=2, V=17)
    if person_num != None: # If a person_num is passed, only get that specific body!
        pose_local = (pose_local[person_num]).reshape((1, 17, 2))


    inner_circ_params = {
        "radius": 5,
        "color": (0, 0, 255) if frame.shape[-1] == 3 else (0, 0, 255, 255),
        "thickness": -1
    }
    outer_circ_params = {
        "radius": 6,
        "color": (255, 0, 0) if frame.shape[-1] == 3 else (255, 0, 0, 255),
        "thickness": 3
    }
    # if frame.shape[1] < 500:
    #     pose_local[:, 0] = pose_local[:, 0] * (319 / 1919)
    #     pose_local[:, 1] = pose_local[:, 1] * (239 / 1079)
    #     circ_params = {"radius": 2, "color": (0, 0, 255), "thickness": 2}
    # Draw the skeleton keypoints on the frame
    for person in pose_local:
        for keypoint_num, keypoint in enumerate(person):
            if 0 in keypoint:
                continue
            if keypoint_num in skip_points:
                continue

            # Draw circle fill first, then the outer circle in blue
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **inner_circ_params)
            cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), **outer_circ_params)

            if debug:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(keypoint_num), (int(keypoint[0]), int(keypoint[1])), font, 0.5,(255,255,255),2,cv2.LINE_AA)
    return frame



def flowpose_lk(frame1, frame2, poses, window_size=3, threshold=0.2, dilation=1, debug_frame=None):
    '''Using the LK method of optical flow calculation...
    CV implementation: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
    goodFeaturesToTrack returns list of length `max_corners`, of shape: [max_corners, 1, 2].
    For each corner, you can simply ravel to flatten the array and get (x,y) positions.
    NOTE: The raw poses (from denoised_skes_data) are of shape: (T, M, V, C)
        In the get_flowpose_samples.py loop, we reshape (poses = poses.transpose(3, 0, 2, 1)) -> (C, T, V, M)

    Args:
        frame1 (torch.Tensor): First frame (grey) of shape (H W)
        frame2 (torch.Tensor): Second frame (grey) of shape (H W)
        poses (torch.Tensor): Pose keypoint tensor of shape ((M V) C)
        window_size (int): The size of the window around each pose keypoint. Default is 3.
        threshold (float): Threshold below which samples are discarded...
        dilation (int): The dilation factor for sampling points around keypoints. Default is 1.
        debug_frame (None/int): Optionally return the frame_number, the frame itself and
            the current state of the flowpose array. Default is None.

    Returns:
        flowpose_aray: Array containing only the flow windows of shape:
            (C*window_size**2, total_keypoints)
    '''
    lk_params = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    half_k = window_size // 2
    pose_local = poses.detach().clone()

    # Get some shapes of input tensors
    height, width = frame1.shape
    total_keypoints, channels = poses.shape

    pose_local[:, 0] = pose_local[:, 0] * (width-1)
    pose_local[:, 1] = pose_local[:, 1] * (height-1)
    pose_local = rearrange(pose_local, '(M V) C -> C (M V)', M=2, V=17)

    # pose_points = ((poses[:2, ...] + 0.5).reshape(2, num_pose_frames, total_keypoints)
    #             * np.array([width - 1, height - 1]).reshape(2, 1, 1)).astype(int)
    vis = pose_local[2, :].flatten() > threshold  # Visibility mask (frames, keypoints)

    # Exclude keypoints that are too close to the edge where the flow window is cut off
    valid_indices = (
        vis.reshape(total_keypoints) &
        (pose_local[0, :] >= half_k * dilation) &
        (pose_local[0, :] < width - half_k * dilation) &
        (pose_local[1, :] >= half_k * dilation) &
        (pose_local[1, :] < height - half_k * dilation)
    )

    # Create the array of just the optical flow windows ((C*H*W), T, V*M)
    flow_windows = np.zeros((window_size**2*2, total_keypoints))

    # Initialise points to track
    p0 = []
    skip_points = []
    for keypoint_num in range(total_keypoints):
        if valid_indices[keypoint_num]:
            x,y = pose_local[0, keypoint_num], pose_local[1, keypoint_num]
            # Create grid of positions about each keypoint ((x,y), 5, 5)
            grid = np.array(
                np.meshgrid(
                    np.linspace(x-half_k*dilation, x+half_k*dilation, window_size).int(),
                    np.linspace(y-half_k*dilation, y+half_k*dilation, window_size).int()
                )
            )
            p0.append(grid)
        else:
            # If keypoint is too close to screen edge...
            p0.append(np.zeros((2, window_size, window_size)))
            skip_points.append(keypoint_num)
            pass
    # Reshape points to track...
    p0 = rearrange(np.array(p0), 'N C H W -> (N H W) 1 C').astype('float32')

    # Estimate the optical flow (LK method)
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

    # Get vectors only for all keypoints on the frame (N=total_keypoints idk why)
    # ((N H W) C) -> ((C H W) N) equivalent to flow_window.flatten
    flow_windows = rearrange(
        (p1-p0).squeeze(),
        '(N H W) C -> (C H W) N',
        N=total_keypoints, H=window_size, W=window_size, C=2
    )
    flow_windows[:, skip_points] = np.zeros((2*(window_size**2), len(skip_points)))

    # Reshape ((C H W) (M V) -> (C H W) V M)
    # get_poses packs the 34 slots person-major, so the unpack must read (M V) too
    # Here, C is the x and y channels of flow, H and W are height and width respectively
    flow_windows = rearrange(flow_windows, 'W (M V) -> W V M', M=2, V=17)
    return flow_windows, p0, p1


def draw_flow_windows(frame, p0, p1, only_middle=False, window_size=3, mag_threshold=20):
    iterator = range((window_size**2)//2, p0.shape[0],(window_size**2)) if only_middle else range(p0.shape[0])
    for point_num in iterator:
        mag = (
            (p1[point_num].ravel()[0]-p0[point_num].ravel()[0])**2 +
            (p1[point_num].ravel()[1]-p0[point_num].ravel()[1])**2)**(0.5)
        if mag > mag_threshold:
            continue
        start = p0[point_num].ravel()
        end = p1[point_num].ravel()
        frame = cv2.arrowedLine(frame, start.astype(int), end.astype(int), (0, 255, 0), 1, tipLength=0.8)
    return frame

def get_Object(frame, object_model, target_classes=None):
    object = object_model.predict(frame, verbose=False)
    for obj in object:
            
            xywh = obj.boxes.xywh#Center ppoint of object
            xywhn = obj.boxes.xywhn #normalized center point of object

            xyxy = obj.boxes.xyxy  #Top left and bottom right corner of object
            xyxyn = obj.boxes.xyxyn #normalized top left and bottom right corner


            if len(obj.boxes) > 0:
                for i, conf in enumerate(obj.boxes.conf):
                    if conf > 0.2:  # Confidence threshold
                        x1 = int(obj.boxes.xyxy[i][0])
                        y1 = int(obj.boxes.xyxy[i][1])
                        x2 = int(obj.boxes.xyxy[i][2])
                        y2 = int(obj.boxes.xyxy[i][3])

                        class_id = int(obj.boxes.cls[i])
                        class_name = obj.names[class_id]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                print("No objects detected")
    return frame
