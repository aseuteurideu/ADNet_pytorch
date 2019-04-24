#ADNet/adnet_test.m
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
import torch
import torch.optim as optim
from models.ADNet import adnet
from options.general import opts
from torch import nn
import torch.utils.data as data
import glob
from datasets.online_adaptation_dataset import OnlineAdaptationDataset, OnlineAdaptationDatasetStorage
from utils.augmentations import ADNet_Augmentation
from utils.do_action import do_action
import time
from utils.display import display_result, draw_box
from utils.gen_samples import gen_samples
from utils.precision_plot import distance_precision_plot, iou_precision_plot
from random import shuffle
from tensorboardX import SummaryWriter

def adnet_test(net, vid_path, opts, args):

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print(
                "WARNING: It looks like you have a CUDA device, but aren't " + "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    transform = ADNet_Augmentation(opts)

    print('Testing sequences in ' + str(vid_path) + '...')
    t_sum = 0

    if args.visualize:
        writer = SummaryWriter(log_dir=os.path.join('tensorboardx_log', 'online_adapatation_' + args.save_result_npy))

    ################################
    # Load video sequences
    ################################

    vid_info={
        'gt' : [],
        'img_files' : [],
        'nframes' : 0
    }

    vid_info['img_files'] = glob.glob(os.path.join(vid_path, 'img', '*.jpg'))
    vid_info['img_files'].sort(key=str.lower)

    gt_path = os.path.join(vid_path, 'groundtruth_rect.txt')

    if not os.path.exists(gt_path):
        bboxes = []
        t = 0
        return bboxes, t_sum

    # parse gt
    gtFile = open(gt_path, 'r')
    gt = gtFile.read().split('\n')
    for i in range(len(gt)):
        if gt[i] == '' or gt[i] is None:
            continue

        if ',' in gt[i]:
            separator = ','
        elif '\t' in gt[i]:
            separator = '\t'
        elif ' ' in gt[i]:
            separator = ' '
        else:
            separator = ','

        gt[i] = gt[i].split(separator)
        gt[i] = list(map(float, gt[i]))
    gtFile.close()

    if len(gt[0]) >= 6:
        for gtidx in range(len(gt)):
            if gt[gtidx] == "":
                continue
            x = gt[gtidx][0:len(gt[gtidx]):2]
            y = gt[gtidx][1:len(gt[gtidx]):2]
            gt[gtidx] = [min(x),
                         min(y),
                         max(x) - min(x),
                         max(y) - min(y)]

    vid_info['gt'] = gt
    if vid_info['gt'][-1] == '':  # small hack
        vid_info['gt'] = vid_info['gt'][:-1]
    vid_info['nframes'] = min(len(vid_info['img_files']), len(vid_info['gt']))

    # catch the first box
    curr_bbox = vid_info['gt'][0]

    # init containers
    bboxes = np.zeros(np.array(vid_info['gt']).shape)  # tracking result containers

    ntraining = 0

    # setup training
    if args.cuda:
        optimizer = optim.SGD([
            {'params': net.module.base_network.parameters(), 'lr': 0},
            {'params': net.module.fc4_5.parameters()},
            {'params': net.module.fc6.parameters()},
            {'params': net.module.fc7.parameters(), 'lr': 1e-3}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])
    else:
        optimizer = optim.SGD([
            {'params': net.base_network.parameters(), 'lr': 0},
            {'params': net.fc4_5.parameters()},
            {'params': net.fc6.parameters()},
            {'params': net.fc7.parameters(), 'lr': 1e-3}],
            lr=1e-3, momentum=opts['train']['momentum'], weight_decay=opts['train']['weightDecay'])

    action_criterion = nn.CrossEntropyLoss()
    score_criterion = nn.CrossEntropyLoss()

    dataset_storage_pos = None
    dataset_storage_neg = None
    is_negative = False  # is_negative = True if the tracking failed
    target_score = 0
    all_iteration = 0
    t = 0

    for idx in range(vid_info['nframes']):
    # for frame_idx, frame_path in enumerate(vid_info['img_files']):
        frame_idx = idx
        frame_path = vid_info['img_files'][idx]
        t0_wholetracking = time.time()
        frame = cv2.imread(frame_path)

        # draw box or with display, then save
        if args.display_images:
            im_with_bb = display_result(frame, curr_bbox)  # draw box and display
        else:
            im_with_bb = draw_box(frame, curr_bbox)

        if args.save_result_images:
            filename = os.path.join(args.save_result_images, str(frame_idx) + '-' + str(t) + '.jpg')
            cv2.imwrite(filename, im_with_bb)

        curr_bbox_old = curr_bbox
        cont_negatives = 0

        if frame_idx > 0:
            # tracking
            if args.cuda:
                net.module.set_phase('test')
            else:
                net.set_phase('test')
            t = 0
            while True:
                curr_patch, curr_bbox, _, _ = transform(frame, curr_bbox, None, None)
                if args.cuda:
                    curr_patch = curr_patch.cuda()

                curr_patch = curr_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]

                fc6_out, fc7_out = net.forward(curr_patch)

                curr_score = fc7_out.detach().cpu().numpy()[0][1]

                if ntraining > args.believe_score_result:
                    if curr_score < opts['failedThre']:
                        cont_negatives += 1

                if args.cuda:
                    action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().cpu().numpy()[0][action]
                else:
                    action = np.argmax(fc6_out.detach().numpy())  # TODO: really okay to detach?
                    action_prob = fc6_out.detach().numpy()[0][action]

                # do action
                curr_bbox = do_action(curr_bbox, opts, action, frame.shape)

                # bound the curr_bbox size
                if curr_bbox[2] < 10:
                    curr_bbox[0] = min(0, curr_bbox[0] + curr_bbox[2] / 2 - 10 / 2)
                    curr_bbox[2] = 10
                if curr_bbox[3] < 10:
                    curr_bbox[1] = min(0, curr_bbox[1] + curr_bbox[3] / 2 - 10 / 2)
                    curr_bbox[3] = 10

                t += 1

                # draw box or with display, then save
                if args.display_images:
                    im_with_bb = display_result(frame, curr_bbox)  # draw box and display
                else:
                    im_with_bb = draw_box(frame, curr_bbox)

                if args.save_result_images:
                    filename = os.path.join(args.save_result_images, str(frame_idx) + '-' + str(t) + '.jpg')
                    cv2.imwrite(filename, im_with_bb)

                if action == opts['stop_action'] or t >= opts['num_action_step_max']:
                    break

            print('final curr_score: %.4f' % curr_score)

            # redetection when confidence < threshold 0.5. But when fc7 is already reliable. Else, just trust the ADNet
            if ntraining > args.believe_score_result:
                if curr_score < 0.5:
                    print('redetection')
                    is_negative = True

                    # redetection process
                    redet_samples = gen_samples('gaussian', curr_bbox_old, opts['redet_samples'], opts, min(1.5, 0.6 * 1.15 ** cont_negatives), opts['redet_scale_factor'])
                    score_samples = []

                    for redet_sample in redet_samples:
                        temp_patch, temp_bbox, _, _ = transform(frame, redet_sample, None, None)
                        if args.cuda:
                            temp_patch = temp_patch.cuda()

                        temp_patch = temp_patch.unsqueeze(0)  # 1 batch input [1, curr_patch.shape]

                        fc6_out_temp, fc7_out_temp = net.forward(temp_patch)

                        score_samples.append(fc7_out_temp.detach().cpu().numpy()[0][1])

                    score_samples = np.array(score_samples)
                    max_score_samples_idx = np.argmax(score_samples)

                    # replace the curr_box with the samples with maximum score
                    curr_bbox = redet_samples[max_score_samples_idx]

                    # update the final result image
                    if args.display_images:
                        im_with_bb = display_result(frame, curr_bbox)  # draw box and display
                    else:
                        im_with_bb = draw_box(frame, curr_bbox)

                    if args.save_result_images:
                        filename = os.path.join(args.save_result_images, str(frame_idx) + '-redet.jpg')
                        cv2.imwrite(filename, im_with_bb)
                else:
                    is_negative = False
            else:
                is_negative = False

        if args.save_result_images:
            filename = os.path.join(args.save_result_images, 'final-' + str(frame_idx) + '.jpg')
            cv2.imwrite(filename, im_with_bb)

        # record the curr_bbox result
        bboxes[frame_idx] = curr_bbox

        # create or update storage + set iteration_range for training
        if frame_idx == 0:
            dataset_storage_pos = OnlineAdaptationDatasetStorage(initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=True)
            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                dataset_storage_neg = OnlineAdaptationDatasetStorage(initial_frame=frame, first_box=curr_bbox, opts=opts, args=args, positive=False)

            iteration_range = range(opts['finetune_iters'])
        else:
            assert dataset_storage_pos is not None
            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                assert dataset_storage_neg is not None

            # if confident or when always generate samples, generate new samples
            if ntraining < args.believe_score_result:
                always_generate_samples = True  # as FC7 wasn't trained, it is better to wait for some time to believe its confidence result to decide whether to generate samples or not.. Before believe it, better to just generate sample always
            else:
                always_generate_samples = False

            if always_generate_samples or (not is_negative or target_score > opts['successThre']):
                dataset_storage_pos.add_frame_then_generate_samples(frame, curr_bbox)

            iteration_range = range(opts['finetune_iters_online'])

        # training when depend on the frequency.. else, don't run the training code...
        if frame_idx % args.online_adaptation_every_I_frames == 0:
            ntraining += 1
            # generate dataset just before training
            dataset_pos = OnlineAdaptationDataset(dataset_storage_pos)
            data_loader_pos = data.DataLoader(dataset_pos, opts['minibatch_size'], num_workers=args.num_workers,
                                              shuffle=True, pin_memory=True)
            batch_iterator_pos = None

            if opts['nNeg_init'] != 0:  # (thanks to small hack in adnet_test) the nNeg_online is also 0
                dataset_neg = OnlineAdaptationDataset(dataset_storage_neg)
                data_loader_neg = data.DataLoader(dataset_neg, opts['minibatch_size'], num_workers=args.num_workers,
                                                  shuffle=True, pin_memory=True)
                batch_iterator_neg = None
            else:
                dataset_neg = []

            epoch_size_pos = len(dataset_pos) // opts['minibatch_size']
            epoch_size_neg = len(dataset_neg) // opts['minibatch_size']
            epoch_size = epoch_size_pos + epoch_size_neg  # 1 epoch, how many iterations

            which_dataset = list(np.full(epoch_size_pos, fill_value=1))
            which_dataset.extend(np.zeros(epoch_size_neg, dtype=int))
            shuffle(which_dataset)

            print("1 epoch = " + str(epoch_size) + " iterations")

            if args.cuda:
                net.module.set_phase('train')
            else:
                net.set_phase('train')

            # training loop
            for iteration in iteration_range:
                all_iteration += 1  # use this for update the visualization
                # create batch iterator
                if (not batch_iterator_pos) or (iteration % epoch_size == 0):
                    batch_iterator_pos = iter(data_loader_pos)

                if opts['nNeg_init'] != 0:
                    if (not batch_iterator_neg) or (iteration % epoch_size == 0):
                        batch_iterator_neg = iter(data_loader_neg)

                # load train data
                if which_dataset[iteration % len(which_dataset)]:  # if positive
                    images, bbox, action_label, score_label = next(batch_iterator_pos)
                else:
                    images, bbox, action_label, score_label = next(batch_iterator_neg)

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

                if all_iteration % 10 == 0:
                    print('Timer: %.4f sec.' % (t1 - t0))
                    print('iter ' + repr(all_iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')
                    if args.visualize and args.send_images_to_visualization:
                        random_batch_index = np.random.randint(images.size(0))
                        writer.add_image('image', images.data[random_batch_index].cpu().numpy(), random_batch_index)

                if args.visualize:
                    writer.add_scalars('data/iter_loss', {'action_loss': action_l.item(),
                                                          'score_loss': score_l.item(),
                                                          'total': (action_l.item() + score_l.item())},
                                       global_step=all_iteration)

        t1_wholetracking = time.time()
        t_sum += t1_wholetracking - t0_wholetracking
        print('whole tracking time = %.4f sec.' % (t1_wholetracking - t0_wholetracking))

    # evaluate the precision
    bboxes = np.array(bboxes)
    vid_info['gt'] = np.array(vid_info['gt'])

    # iou_precisions = iou_precision_plot(bboxes, vid_info['gt'], vid_path, show=args.display_images, save_plot=args.save_result_images)
    #
    # distance_precisions = distance_precision_plot(bboxes, vid_info['gt'], vid_path, show=args.display_images, save_plot=args.save_result_images)
    #
    # precisions = [distance_precisions, iou_precisions]

    np.save(args.save_result_npy + '-bboxes.npy', bboxes)
    np.save(args.save_result_npy + '-ground_truth.npy', vid_info['gt'])

    # return bboxes, t_sum, precisions
    return bboxes, t_sum
