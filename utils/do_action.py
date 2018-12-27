import numpy as np

def do_action(bbox, opts, act, imSize):
    m = opts['action_move']

    # action
    bbox[0] = bbox[0] + 0.5 * bbox[2]
    bbox[1] = bbox[1] + 0.5 * bbox[3]

    deltas = [m['x'] * bbox[2],
              m['y'] * bbox[3],
              m['w'] * bbox[2],
              m['h'] * bbox[3]]

    deltas = np.maximum(deltas, 1)

    ar = bbox[2]/bbox[3]

    if bbox[2] > bbox[3]:
        deltas[3] = deltas[2] / ar

    else:
        deltas[2] = deltas[3] * ar

    action_delta = np.multiply(np.array(m['deltas'])[act, :], deltas)
    bbox_next = bbox + action_delta
    bbox_next[0] = bbox_next[0] - 0.5 * bbox_next[2]
    bbox_next[1] = bbox_next[1] - 0.5 * bbox_next[3]
    bbox_next[0] = np.maximum(bbox_next[0], 1)
    bbox_next[0] = np.minimum(bbox_next[0], imSize[1] - bbox_next[2])
    bbox_next[1] = np.maximum(bbox_next[1], 1)
    bbox_next[1] = np.minimum(bbox_next[1], imSize[0] - bbox_next[3])
    bbox_next[2] = np.maximum(5, np.minimum(imSize[1], bbox_next[2]))
    bbox_next[3] = np.maximum(5, np.minimum(imSize[0], bbox_next[3]))

    bbox[0] = bbox[0] - 0.5 * bbox[2]
    bbox[1] = bbox[1] - 0.5 * bbox[3]

    return bbox_next


# # test module
# from utils.init_params import opts
# bbox = [11, 12, 13, 14]
# imSize = [112, 112, 3]
# act = 2
# new_bbox = do_action(bbox, opts, act, imSize)