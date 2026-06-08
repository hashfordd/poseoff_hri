import argparse
import cv2
from utils import *
import time
import numpy as np
import pyrealsense2 as rs

def main(pose_model, im_height, im_width):
    print("\n ------- PRESS `Q` TO QUIT ------ \n")
    #Start Camera
    pipeline.start(config)
    
    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()

    img1 = np.asanyarray(color_frame.get_data())
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    try:
        while True:
            #Start Camera
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            img2 = np.asanyarray(color_frame.get_data())
            img3 = np.asanyarray(color_frame.get_data())

            poses = get_poses(img2, pose_model, threshold=threshold)
            img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            poseoff, p0, p1 = flowpose_lk(img1_grey, img2_grey, poses, window_size=window_size, dilation=dilation)
            
            #img2 = get_Object(img2, object_model, target_classes)

            img2 = draw_bones(img2, poses)
            img2 = draw_flow_windows(img2, p0, p1, only_middle=False, window_size=window_size)

    

            #needed if dropping resolution
            img2 = cv2.resize(img2, (im_width*2, im_height*2))

            cv2.imshow("RealSense D456 Viewer", img2)


            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command line arguments
    threshold = 0.2
    window_size = 5
    dilation = 3

    im_height = 270
    im_width = 480


    # Added for object detection
    target_classes = ["ball", 'cup', 'bottle', 'bannana', 'apple']

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, im_width , im_height, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, im_width, im_height, rs.format.z16, 30)

    # Create YOLO-pose model
    pose_model = YOLO("yolo26m-pose.pt")

    object_model = YOLO('yoloe-26n-seg.pt')
    object_model.set_classes(target_classes)

    main(pose_model=pose_model, im_height = im_height , im_width =im_width)