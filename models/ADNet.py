from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.legacy import nn as nnl
import torch.utils.model_zoo as model_zoo

import numpy as np
import os

from utils.get_action_history_onehot import get_action_history_onehot

from models.vggm import vggm

__all__ = ['vggm']

pretrained_settings = {
    'adnet': {
        'input_space': 'BGR',
        'input_size': [3, 112, 112],
        'input_range': [0, 255],
        'mean': [123.68, 116.779, 103.939],
        'std': [1, 1, 1],
        'num_classes': 11
    }
}


class ADNet(nn.Module):

    def __init__(self, base_network, opts, num_classes=11, phase='train', num_history=10):
        super(ADNet, self).__init__()

        self.num_classes = num_classes
        self.phase = phase
        self.opts = opts

        self.base_network = base_network
        self.fc4_5 = nn.Sequential(
            nn.Linear(18432, 512), # TODO: (slightly different from paper): nn.Linear(4608, 512),  # [0]
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),  # [3]
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -1 to differentiate between action '0' and haven't been explored
        self.action_history = np.full(num_history, -1)

        self.action_dynamic_size = num_classes * num_history
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size))

        # TODO: current assumption: still always with cuda
        self.action_dynamic = self.action_dynamic.cuda()

        self.fc6 = nn.Linear(512 + self.action_dynamic_size, self.num_classes)

        self.fc7 = nn.Linear(512 + self.action_dynamic_size, 2)

        self.softmax = nn.Softmax()

    # update_action_dynamic: history of action. We don't update the action_dynamic in SL learning.
    def forward(self, x, action_dynamic=None, update_action_dynamic=False):
        assert x is not None
        x = self.base_network(x)
        x = x.view(x.size(0), -1)
        x = self.fc4_5(x)

        if action_dynamic is None:
            x = torch.cat((x, self.action_dynamic.expand(x.shape[0], self.action_dynamic.shape[0])), 1)
        else:
            x = torch.cat((x, action_dynamic))

        fc6_out = self.fc6(x)
        fc7_out = self.fc7(x)

        if self.phase == 'test':
            fc6_out = self.softmax(fc6_out)
            fc7_out = self.softmax(fc7_out)

        if update_action_dynamic:
            selected_action = np.argmax(fc6_out.detach().cpu().numpy())  # TODO: really okay to detach?
            self.action_history[1:] = self.action_history[0:-1]
            self.action_history[0] = selected_action
            self.update_action_dynamic(self.action_history)

        return fc6_out, fc7_out

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def update_action_dynamic(self, action_history):
        onehot_action = get_action_history_onehot(action_history, self.opts)

        action_dynamic = onehot_action

        self.action_dynamic = torch.Tensor(np.array(action_dynamic)).cuda()  # TODO: current assumption: still always with cuda

    def reset_action_dynamic(self):
        self.action_dynamic = torch.Tensor(np.zeros(self.action_dynamic_size))

    def get_action_dynamic(self):
        return self.action_dynamic

    def set_phase(self, phase):
        self.phase = phase


def adnet(opts, base_network='vggm', trained_file=None):
    assert base_network in ['vggm'], "Base network variant is unavailable"

    num_classes = opts['num_actions']
    num_history = opts['num_action_history']
    assert num_classes in [11, 13], "num classes is not exist"

    if num_classes == 11:
        settings = pretrained_settings['adnet']
    else:  # elif num_classes == 13:
        settings = pretrained_settings['adnet_13_action']

    if base_network == 'vggm':
        base_network = vggm()  # by default, load vggm's weights too
        base_network = base_network.features[0:10]

    else:  # change this part if adding more base network variant
        base_network = vggm()
        base_network = base_network.features[0:10]

    if trained_file:
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes, num_history=num_history)
        model.load_state_dict(torch.load(trained_file, map_location=lambda storage, loc: storage))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = ADNet(base_network=base_network, opts=opts, num_classes=num_classes)
    return model