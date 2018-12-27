# pytorch dataset for SL learning
# matlab code (line 26-33):
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference:
# https://github.com/amdegroot/ssd.pytorch/blob/master/data/voc0712.py

import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from datasets.get_train_dbs import get_train_dbs
from utils.get_video_infos import get_video_infos


class SLDataset(data.Dataset):
    # train_videos = get_train_videos(opts)
    # train_videos = {  # the format of train_videos
    #         'video_names' : video_names,
    #         'video_paths' : video_paths,
    #         'bench_names' : bench_names
    #     }
    def __init__(self, train_db, transform=None):
        self.transform = transform
        self.train_db = train_db

    def __getitem__(self, index):
        im = cv2.imread(self.train_db['img_path'][index])
        bbox = self.train_db['bboxes'][index]
        action_label = np.array(self.train_db['labels'][index], dtype=np.float32)
        score_label = self.train_db['score_labels'][index]

        if self.transform is not None:
            im, bbox, action_label, score_label = self.transform(im, bbox, action_label, score_label)

        return im, bbox, action_label, score_label

    def __len__(self):
        return len(self.train_db['img_path'])

    #########################################################
    # ADDITIONAL FUNCTIONS

    def pull_image(self, index):
        im = cv2.imread(self.train_db['img_path'][index])
        return im

    def pull_anno(self, index):
        action_label = self.train_db['labels'][index]
        score_label = self.train_db['score_labels'][index]
        return action_label, score_label

def initialize_pos_neg_dataset(train_videos, opts, transform=None):
    num_videos = len(train_videos['video_names'])

    train_db_pos = {
        'img_path': [],  # list of string
        'bboxes': [],  # list of ndarray left top coordinate [left top width height]
        'labels': [],  # list of ndarray #action elements. One hot vector
        'score_labels': []  # list of scalar 0 (negative) or 1 (positive)
    }
    train_db_neg = {
        'img_path': [],  # list of string
        'bboxes': [],  # list of ndarray left top coordinate [left top width height]
        'labels': [],  # list of ndarray #action elements. One hot vector
        'score_labels': []  # list of scalar 0 (negative) or 1 (positive)
    }

    for vid_idx in range(num_videos):
        print("generating dataset from video " + str(vid_idx + 1) + "/" + str(num_videos) +
              "(current total data (pos-neg): " + str(len(train_db_pos['labels'])) +
              "-" + str(len(train_db_neg['labels'])) + ")")

        bench_name = train_videos['bench_names'][vid_idx]
        video_name = train_videos['video_names'][vid_idx]
        video_path = train_videos['video_paths'][vid_idx]
        vid_info = get_video_infos(bench_name, video_path, video_name)
        train_db_pos_, train_db_neg_ = get_train_dbs(vid_info, opts)
        # separate for each bboxes sample
        for sample_idx in range(len(train_db_pos_)):
            # for img_path_idx in range(len(train_db_pos_[sample_idx]['score_labels'])):
            train_db_pos['img_path'].extend(train_db_pos_[sample_idx]['img_path'])
            train_db_pos['bboxes'].extend(train_db_pos_[sample_idx]['bboxes'])
            train_db_pos['labels'].extend(train_db_pos_[sample_idx]['labels'])
            train_db_pos['score_labels'].extend(train_db_pos_[sample_idx]['score_labels'])

        print("Finish generating positive dataset... (current total data: " + str(len(train_db_pos['labels'])) + ")")

        for sample_idx in range(len(train_db_neg_)):
            # for img_path_idx in range(len(train_db_neg_[sample_idx]['score_labels'])):
            train_db_neg['img_path'].extend(train_db_neg_[sample_idx]['img_path'])
            train_db_neg['bboxes'].extend(train_db_neg_[sample_idx]['bboxes'])
            train_db_neg['labels'].extend(train_db_neg_[sample_idx]['labels'])
            train_db_neg['score_labels'].extend(train_db_neg_[sample_idx]['score_labels'])

        print("Finish generating negative dataset... (current total data: " + str(len(train_db_neg['labels'])) + ")")

    dataset_pos = SLDataset(train_db_pos, transform=transform)
    dataset_neg = SLDataset(train_db_neg, transform=transform)

    return dataset_pos, dataset_neg
