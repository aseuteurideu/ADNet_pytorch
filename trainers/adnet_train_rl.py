# matlab code: https://github.com/hellbell/ADNet/blob/master/train/adnet_train_RL.m
# policy gradient in pytorch: https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import numpy as np
from utils.get_video_infos import get_video_infos
import cv2
from utils.augmentations import ADNet_Augmentation
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
from trainers.RL_tools import TrackingEnvironment
import torch.optim as optim
from trainers.RL_tools import TrackingPolicyLoss
from torch.distributions import Categorical
from utils.display import display_result
import copy
from datasets.rl_dataset import RLDataset
import torch.utils.data as data
from tensorboardX import SummaryWriter

def adnet_train_rl(net, train_videos, opts, args):
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.visualize:
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', args.save_file_RL))

    if args.run_supervised:  # if already run the supervised, the net variable has been processed
        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            state_dict = torch.load(args.resume)
            net.load_state_dict(state_dict)

        if args.cuda:
            net = nn.DataParallel(net)
            cudnn.benchmark = True

        if args.cuda:
            net = net.cuda()

    if args.cuda:
        net.module.set_phase('test')

        # I just make the learning rate same with SL, except that we don't train the FC7
        optimizer = optim.SGD([
            {'params': net.module.base_network.parameters(), 'lr': 1e-4},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters(), 'lr': 0}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    else:
        net.set_phase('test')

        # I just make the learning rate same with SL, except that we don't train the FC7
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 1e-4},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters(), 'lr': 0}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    criterion = TrackingPolicyLoss()

    ###########################################################
    clip_idx_epoch = 0
    prev_net = copy.deepcopy(net)
    dataset = RLDataset(prev_net, train_videos, opts, args)

    for epoch in range(args.start_epoch, opts['numEpoch']):
        if epoch != args.start_epoch:
            prev_net = copy.deepcopy(net)  # save the not updated net for generating data
            dataset.reset(prev_net, train_videos, opts, args)

        data_loader = data.DataLoader(dataset, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True,
                                      pin_memory=True)

        # create batch iterator
        batch_iterator = iter(data_loader)

        epoch_size = len(dataset) // opts['minibatch_size']   # 1 epoch, how many iterations

        total_loss_epoch = 0
        total_reward_epoch = 0

        for iteration in range(epoch_size):
            # load train data
            # action, action_prob, log_probs, reward, patch, action_dynamic, result_box = next(batch_iterator)
            log_probs, reward = next(batch_iterator)

            # train
            tic = time.time()

            optimizer.zero_grad()
            # loss = criterion(tracking_scores, num_frame, num_step_history, action_prob_history)
            loss = criterion(log_probs, reward)
            loss.backward()
            reward_sum = reward.sum()

            optimizer.step()  # update

            toc = time.time() - tic
            print('epoch ' + str(epoch) + ' - iteration ' + str(iteration) + ' - train time: ' + str(toc) + " s")

            if args.visualize:
                writer.add_scalar('data/iter_reward_sum', reward_sum, iteration)
                writer.add_scalar('data/iter_loss', loss, iteration)

            if iteration % 1000 == 0:
                torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file_RL) +
                           '_epoch' + repr(epoch) + '_iter' + repr(iteration) +'.pth')

            total_loss_epoch += loss
            total_reward_epoch += reward_sum
            clip_idx_epoch += 1

        if args.visualize:
            writer.add_scalar('data/epoch_reward_ave', total_reward_epoch/epoch_size, epoch)
            writer.add_scalar('data/epoch_loss', total_loss_epoch/epoch_size, epoch)

        torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file_RL) +
                   'epoch' + repr(epoch) + '.pth')

    torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file_RL) + '.pth')

    return net


# # test the module
# from models.ADNet import adnet
# from utils.get_train_videos import get_train_videos
# from options.general import opts
# import argparse
#
# def str2bool(v):
#     return v.lower() in ("yes", "true", "t", "1")
#
# parser = argparse.ArgumentParser(
#     description='ADNet training')
# parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
# parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
# parser.add_argument('--visdom', default=True, type=str2bool, help='Use visdom to for loss visualization')
# parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
# parser.add_argument('--save_folder', default='weights', help='Location to save checkpoint models')
#
# parser.add_argument('--save_file', default='ADNet_RL_', type=str, help='save file part of file name')
# parser.add_argument('--start_epoch', default=0, type=int, help='Begin counting epochs starting from this value')
#
# parser.add_argument('--run_supervised', default=True, type=str2bool, help='Whether to run supervised learning or not')
#
# args = parser.parse_args()
#
# opts['minibatch_size'] = 32  # TODO: still don't know what this parameter for....
#
# net = adnet(opts)
# train_videos = get_train_videos(opts)
# adnet_train_rl(net, train_videos, opts, args)
