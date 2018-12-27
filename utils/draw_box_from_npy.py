import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import glob
import os
import cv2
from utils.display import draw_box


def draw_box_from_npy(video_path, npy_file, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    bboxes = np.load(npy_file)

    frames_files = os.listdir(video_path)
    frames_files.sort(key=str.lower)

    for frame_idx, frame_file in enumerate(frames_files):
        frame = cv2.imread(os.path.join(video_path, frame_file))
        curr_bbox = bboxes[frame_idx]
        im_with_bb = draw_box(frame, curr_bbox)

        filename = os.path.join(save_path, str(frame_idx) + '.jpg')
        cv2.imwrite(filename, im_with_bb)


#test the module
draw_box_from_npy('/home/astrid/RL_class/ADNet-pytorch/datasets/data/otb/Basketball/img', '/home/astrid/RL_class/ADNet-pytorch/mains/results_on_test_images_part2/ADNet_RL_-0.5/Basketball-bboxes.npy', '/home/astrid/RL_class/ADNet-pytorch/mains/results_on_test_images_part2/ADNet_RL_-0.5/Basketball')