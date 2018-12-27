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

import time
from trainers.RL_tools import TrackingEnvironment
from utils.augmentations import ADNet_Augmentation, ADNet_Augmentation_KeepAspectRatio
from utils.display import display_result, draw_box
from torch.distributions import Categorical

class RLDataset(data.Dataset):

    def __init__(self, net, train_videos, opts, args):
        self.env = None

        # these lists won't include the ground truth
        self.action_list = []  # a_t,l  # argmax of self.action_prob_list
        self.action_prob_list = []  # output of network (fc6_out)
        self.log_probs_list = []  # log probs from each self.action_prob_list member
        self.reward_list = []  # tracking score
        self.patch_list = []  # input of network
        self.action_dynamic_list = []  # action_dynamic used for inference (means before updating the action_dynamic)
        self.result_box_list = []

        self.reset(net, train_videos, opts, args)

    def __getitem__(self, index):

        # return self.action_list[index], self.action_prob_list[index], self.log_probs_list[index], \
        #        self.reward_list[index], self.patch_list[index], self.action_dynamic_list[index], \
        #        self.result_box_list[index]

        # TODO: currently only log_probs_list and reward_list contains data
        return self.log_probs_list[index], self.reward_list[index]

    def __len__(self):
        return len(self.log_probs_list)

    def reset(self, net, train_videos, opts, args):
        print('generating reinforcement learning dataset')
        transform = ADNet_Augmentation(opts)

        self.env = TrackingEnvironment(train_videos, opts, transform=transform, args=args)
        clip_idx = 0
        while True:  # for every clip (l)

            num_step_history = [1]  # T_l

            num_frame = 1  # the first frame won't be tracked..
            t = 0
            box_history_clip = []  # for checking oscillation in a clip

            if args.cuda:
                net.module.reset_action_dynamic()
            else:
                net.reset_action_dynamic()  # action dynamic should be in a clip (what makes sense...)

            while True:  # for every frame in a clip (t)
                tic = time.time()

                if args.display_images:
                    im_with_bb = display_result(self.env.get_current_img(), self.env.get_state())
                    cv2.imshow('patch', self.env.get_current_patch_unprocessed())
                    cv2.waitKey(1)
                else:
                    im_with_bb = draw_box(self.env.get_current_img(), self.env.get_state())

                if args.save_result_images:
                    cv2.imwrite('images/' + str(clip_idx) + '-' + str(t) + '.jpg', im_with_bb)

                curr_patch = self.env.get_current_patch()
                if args.cuda:
                    curr_patch = curr_patch.cuda()

                # self.patch_list.append(curr_patch.cpu().data.numpy())  # TODO: saving patch takes cuda memory

                # TODO: saving action_dynamic takes cuda memory
                # if args.cuda:
                #     self.action_dynamic_list.append(net.module.get_action_dynamic())
                # else:
                #     self.action_dynamic_list.append(net.get_action_dynamic())

                curr_patch = curr_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]

                fc6_out, fc7_out = net.forward(curr_patch, update_action_dynamic=True)

                if args.cuda:
                    action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().cpu().numpy()[0][action]
                else:
                    action = np.argmax(fc6_out.detach().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().numpy()[0][action]

                m = Categorical(probs=fc6_out)
                action_ = m.sample()  # action and action_ are same value. Only differ in the type (int and tensor)

                self.log_probs_list.append(m.log_prob(action_).cpu().data.numpy())

                self.action_list.append(action)
                # TODO: saving action_prob_list takes cuda memory
                # self.action_prob_list.append(action_prob)

                new_state, reward, done, info = self.env.step(action)

                # check oscilating
                if any((np.array(new_state).round() == x).all() for x in np.array(box_history_clip).round()):
                    action = opts['stop_action']
                    reward, done, finish_epoch = self.env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # check if number of action is already too much
                if t >= opts['num_action_step_max'] - 1:
                    action = opts['stop_action']
                    reward, done, finish_epoch = self.env.go_to_next_frame()
                    info['finish_epoch'] = finish_epoch

                # TODO: saving result_box takes cuda memory
                # self.result_box_list.append(list(new_state))
                box_history_clip.append(list(new_state))

                t += 1

                if action == opts['stop_action']:
                    num_frame += 1
                    num_step_history.append(t)
                    t = 0

                toc = time.time() - tic
                print('forward time (clip ' + str(clip_idx) + " - frame " + str(num_frame) + " - t " + str(t) + ") = "
                      + str(toc) + " s")

                if done:  # if finish the clip
                    break

            tracking_scores_size = np.array(num_step_history).sum()
            tracking_scores = np.full(tracking_scores_size, reward)  # seems no discount factor whatsoever

            self.reward_list.extend(tracking_scores)

            clip_idx += 1

            if info['finish_epoch']:
                break

        print('generating reinforcement learning dataset finish')

