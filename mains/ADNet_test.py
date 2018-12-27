import argparse
from options.general import opts
from models.ADNet import adnet
from utils.get_train_videos import get_train_videos
from trainers.adnet_test import adnet_test
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os
import glob

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='ADNet test')
parser.add_argument('--weight_file', default='weights/ADNet_RL_.pth', type=str, help='The pretrained weight file')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--visualize', default=False, type=str2bool, help='Use tensorboardx to for visualization')
parser.add_argument('--send_images_to_visualization', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--display_images', default=True, type=str2bool, help='Whether to display images or not')
parser.add_argument('--save_result_images', default=None, type=str, help='save results folder')
parser.add_argument('--save_result_npy', default='results_on_test_images_part2', type=str, help='save results folder')

parser.add_argument('--initial_samples', default=3000, type=int, help='Number of training samples for the first frame. N_I')
parser.add_argument('--online_samples', default=250, type=int, help='Number of training samples for the other frames. N_O')
parser.add_argument('--redetection_samples', default=256, type=int, help='Number of samples for redetection. N_det')
parser.add_argument('--initial_iteration', default=300, type=int, help='Number of iteration in initial training. T_I')
parser.add_argument('--online_iteration', default=30, type=int, help='Number of iteration in online training. T_O')
parser.add_argument('--online_adaptation_every_I_frames', default=10, type=int, help='Frequency of online training. I')
# parser.add_argument('--number_past_frames', default=20, type=int, help='The training data were sampled from the past J frames. J')

parser.add_argument('--believe_score_result', default=0, type=int, help='Believe score result after n training')

parser.add_argument('--pos_samples_ratio', default='0.5', type=float,
                    help='The ratio of positive in all samples for online adaptation. Rest of it will be negative samples. Default: 0.5')


args = parser.parse_args()
assert 0 < args.pos_samples_ratio <= 1, "the pos_samples_ratio valid range is (0, 1]"

# set opts based on the args.. especially the number of samples etc.
opts['nPos_init'] = int(args.initial_samples * args.pos_samples_ratio)
opts['nNeg_init'] = int(args.initial_samples - opts['nPos_init'])
opts['nPos_online'] = int(args.online_samples * args.pos_samples_ratio)
opts['nNeg_online'] = int(args.online_samples - opts['nPos_online'])

# just to make sure if one of nNeg is zero, the other nNeg is zero (kinda small hack...)
if opts['nNeg_init'] == 0:
    opts['nNeg_online'] = 0
    opts['nPos_online'] = args.online_samples

elif opts['nNeg_online'] == 0:
    opts['nNeg_init'] = 0
    opts['nPos_init'] = args.initial_samples

opts['finetune_iters'] = args.initial_iteration
opts['finetune_iters_online'] = args.online_iteration
opts['redet_samples'] = args.redetection_samples

if args.save_result_images is not None:
    args.save_result_images = os.path.join(args.save_result_images,
                                           os.path.basename(args.weight_file)[:-4] + '-' + str(args.pos_samples_ratio))
    if not os.path.exists(args.save_result_images):
        os.mkdir(args.save_result_images)

args.save_result_npy = os.path.join(args.save_result_npy, os.path.basename(args.weight_file)[:-4] + '-' +
                                    str(args.pos_samples_ratio))
if not os.path.exists(args.save_result_npy):
    os.mkdir(args.save_result_npy)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print(
            "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

net = adnet(opts)
net.train()
if args.cuda:
    net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.cuda:
    net = net.cuda()

dataset_root = os.path.join('../datasets/data', opts['test_db'])
vid_folders = []
for filename in os.listdir(dataset_root):
    if os.path.isdir(os.path.join(dataset_root,filename)):
        vid_folders.append(filename)
vid_folders.sort(key=str.lower)
# all_precisions = []

save_root = args.save_result_images
save_root_npy = args.save_result_npy

for vid_folder in vid_folders:
    print('Loading {}...'.format(args.weight_file))
    state_dict = torch.load(args.weight_file)
    net.load_state_dict(state_dict)

    if args.save_result_images is not None:
        args.save_result_images = os.path.join(save_root, vid_folder)
        if not os.path.exists(args.save_result_images):
            os.mkdir(args.save_result_images)

    args.save_result_npy = os.path.join(save_root_npy, vid_folder)

    vid_path = os.path.join(dataset_root, vid_folder)

    bboxes, t_sum = adnet_test(net, vid_path, opts, args)
#     all_precisions.append(precisions)
#
# print(all_precisions)
