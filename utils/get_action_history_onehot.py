# returns action history as one-hot form
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/get_action_history_onehot.m
import numpy as np

def get_action_history_onehot(action_history, opts):
    ah_onehot = []
    for i in range(len(action_history)):
        onehot = np.zeros(opts['num_actions'])
        if action_history[i] >= 0 and action_history[i] < opts['num_actions']:
            onehot[action_history[i]] = 1

        ah_onehot.extend(onehot)

    return ah_onehot

# test the module
# from utils.init_params import opts
# action_history = np.array([0,2,4,2,-1,-1,-1])
# get_action_history_onehot(action_history, opts)
