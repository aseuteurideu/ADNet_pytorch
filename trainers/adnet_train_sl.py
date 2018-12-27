# matlab code:
# https://github.com/hellbell/ADNet/blob/master/train/adnet_train_SL.m
# reference: https://github.com/amdegroot/ssd.pytorch/blob/master/train.py

import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from models.ADNet import adnet
from utils.get_train_videos import get_train_videos
from datasets.sl_dataset import initialize_pos_neg_dataset
from utils.augmentations import ADNet_Augmentation

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from random import shuffle

import os
import time
import numpy as np

from tensorboardX import SummaryWriter


def adnet_train_sl(args, opts):

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
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', args.save_file))

    net = adnet(opts=opts)

    if args.cuda:
        net = nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        # net.load_weights(args.resume)
        state_dict = torch.load(args.resume)
        net.load_state_dict(state_dict)

    if args.cuda:
        net = net.cuda()

    net.train()
    action_loss = 0
    score_loss = 0


    if not args.resume:
        print('Initializing weights...')

        if args.cuda:
            scal = torch.Tensor([0.01])
            # fc 4
            nn.init.normal_(net.module.fc4_5[0].weight.data)
            net.module.fc4_5[0].weight.data = net.module.fc4_5[0].weight.data * scal.expand_as(net.module.fc4_5[0].weight.data)
            net.module.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.module.fc4_5[3].weight.data)
            net.module.fc4_5[3].weight.data = net.module.fc4_5[3].weight.data * scal.expand_as(net.module.fc4_5[3].weight.data)
            net.module.fc4_5[3].bias.data.fill_(0.1)
            # fc 6
            nn.init.normal_(net.module.fc6.weight.data)
            net.module.fc6.weight.data = net.module.fc6.weight.data * scal.expand_as(net.module.fc6.weight.data)
            net.module.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.module.fc7.weight.data)
            net.module.fc7.weight.data = net.module.fc7.weight.data * scal.expand_as(net.module.fc7.weight.data)
            net.module.fc7.bias.data.fill_(0)
        else:
            scal = torch.Tensor([0.01])
            # fc 4
            nn.init.normal_(net.fc4_5[0].weight.data)
            net.fc4_5[0].weight.data = net.fc4_5[0].weight.data * scal.expand_as(net.fc4_5[0].weight.data )
            net.fc4_5[0].bias.data.fill_(0.1)
            # fc 5
            nn.init.normal_(net.fc4_5[3].weight.data)
            net.fc4_5[3].weight.data = net.fc4_5[3].weight.data * scal.expand_as(net.fc4_5[3].weight.data)
            net.fc4_5[3].bias.data.fill_(0.1)
            # fc 6
            nn.init.normal_(net.fc6.weight.data)
            net.fc6.weight.data = net.fc6.weight.data * scal.expand_as(net.fc6.weight.data)
            net.fc6.bias.data.fill_(0)
            # fc 7
            nn.init.normal_(net.fc7.weight.data)
            net.fc7.weight.data = net.fc7.weight.data * scal.expand_as(net.fc7.weight.data)
            net.fc7.bias.data.fill_(0)

    if args.cuda:
        optimizer = optim.SGD([
            {'params': net.module.base_network.parameters(), 'lr': 1e-4},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters()}],  # as action dynamic is zero, it doesn't matter
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    else:
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 1e-4},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters()}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.CrossEntropyLoss()

    train_videos = get_train_videos(opts)
    opts['num_videos'] = len(train_videos['video_names'])

    print('generating Supervised Learning dataset..')
    # dataset = SLDataset(train_videos, opts, transform=ADNet_Augmentation(opts))
    dataset_pos, dataset_neg = initialize_pos_neg_dataset(train_videos, opts, transform=ADNet_Augmentation(opts))

    batch_iterator_pos = None
    batch_iterator_neg = None

    epoch_size_pos = len(dataset_pos) // opts['minibatch_size']
    epoch_size_neg = len(dataset_neg) // opts['minibatch_size']
    epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations
    print("1 epoch = " + str(epoch_size) + " iterations")

    max_iter = opts['numEpoch'] * epoch_size
    print("maximum iteration = " + str(max_iter))

    data_loader_pos = data.DataLoader(dataset_pos, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True, pin_memory=True)
    data_loader_neg = data.DataLoader(dataset_neg, opts['minibatch_size'], num_workers=args.num_workers, shuffle=True, pin_memory=True)

    epoch = args.start_epoch
    if epoch != 0 and args.start_iter == 0:
        start_iter = epoch * epoch_size
    else:
        start_iter = args.start_iter

    which_dataset = list(np.full(epoch_size_pos, fill_value=1))
    which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
    shuffle(which_dataset)

    # training loop
    for iteration in range(start_iter, max_iter):
        if (iteration != start_iter) and (iteration % epoch_size == 0):
            epoch += 1
            shuffle(which_dataset)

            print('Saving state, epoch:', epoch)
            torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file) +
                       'epoch' + repr(epoch) + '.pth')

            if args.visualize:
                writer.add_scalars('data/epoch_loss', {'action_loss': action_loss / epoch_size,
                                                       'score_loss': score_loss / epoch_size,
                                                       'total': (action_loss + score_loss) / epoch_size}, global_step=epoch)

            # reset epoch loss counters
            action_loss = 0
            score_loss = 0

        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator_pos = iter(data_loader_pos)
            batch_iterator_neg = iter(data_loader_neg)

        if not batch_iterator_pos:
            # create batch iterator
            batch_iterator_pos = iter(data_loader_pos)

        if not batch_iterator_neg:
            # create batch iterator
            batch_iterator_neg = iter(data_loader_neg)


        # load train data
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            images, bbox, action_label, score_label = next(batch_iterator_pos)
        else:
            images, bbox, action_label, score_label = next(batch_iterator_neg)

        # TODO: check if this requires grad is really false like in Variable
        if args.cuda:
            images = torch.Tensor(images.cuda())
            bbox = torch.Tensor(bbox.cuda())
            action_label = torch.Tensor(action_label.cuda())
            score_label = torch.Tensor(score_label.float().cuda())

        else:
            images = torch.Tensor(images)
            bbox = torch.Tensor(bbox)
            action_label = torch.Tensor(action_label)
            score_label = torch.Tensor(score_label)

        # forward
        t0 = time.time()
        action_out, score_out = net(images)

        # backprop
        optimizer.zero_grad()
        if which_dataset[iteration % len(which_dataset)]:  # if positive
            action_l = action_criterion(action_out, torch.max(action_label, 1)[1])
        else:
            action_l = torch.Tensor([0])
        score_l = score_criterion(score_out, score_label.long())
        loss = action_l + score_l
        loss.backward()
        optimizer.step()
        t1 = time.time()
        action_loss += action_l.item()
        score_loss += score_l.item()

        if iteration % 10 == 0:
            print('Timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
            if args.visualize and args.send_images_to_visualization:
                random_batch_index = np.random.randint(images.size(0))
                writer.add_image('image', images.data[random_batch_index].cpu().numpy(), random_batch_index)

        if args.visualize:
            writer.add_scalars('data/iter_loss', {'action_loss': action_l.item(),
                                                  'score_loss': score_l.item(),
                                                  'total': (action_l.item() + score_l.item())}, global_step=iteration)
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                writer.add_scalars('data/epoch_loss', {'action_loss': action_loss,
                                                       'score_loss': score_loss,
                                                       'total': (action_loss + score_loss)}, global_step=epoch)

        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file) +
                       repr(iteration) + '_epoch' + repr(epoch) +'.pth')

    torch.save(net.state_dict(), os.path.join(args.save_folder, args.save_file) + '.pth')

    return net, train_videos



