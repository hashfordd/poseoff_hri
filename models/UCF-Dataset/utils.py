import torch
from einops import rearrange
import numpy as np
import cv2

def get_poses(frame, pose_model, threshold=0.2):
    results = pose_model(frame, device='cuda', verbose=False)
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
