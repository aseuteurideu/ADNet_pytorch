# matlab source:
# https://github.com/hellbell/ADNet/blob/3a7955587b5d395401ebc94a5ab067759340680d/utils/overlap_ratio.m

import numpy as np


# def rectint(a, b):  # returns 0 if rectangles don't intersect
#     assert (isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray))) or \
#            (not isinstance(a, (list, np.ndarray)) and not isinstance(b, (list, np.ndarray)))
#
#     if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)):
#         results = []
#         for _a, _b in zip(a, b):
#             _a_xmin = _a[0]
#             _a_ymin = _a[1]
#             _a_xmax = _a[0] + _a[2]
#             _a_ymax = _a[1] + _a[3]
#
#             _b_xmin = _b[0]
#             _b_ymin = _b[1]
#             _b_xmax = _b[0] + _b[2]
#             _b_ymax = _b[1] + _b[3]
#
#             dx = min(_a_xmax, _b_xmax) - max(_a_xmin, _b_xmin)
#             dy = min(_a_ymax, _b_ymax) - max(_a_ymin, _b_ymin)
#
#             if (dx >= 0) and (dy >= 0):
#                 results.append(dx * dy)
#             else:
#                 results.append(0)
#
#         return results
#
#     else:
#         a_xmin = a[0]
#         a_ymin = a[1]
#         a_xmax = a[0] + a[2]
#         a_ymax = a[1] + a[3]
#
#         b_xmin = b[0]
#         b_ymin = b[1]
#         b_xmax = b[0] + b[2]
#         b_ymax = b[1] + b[3]
#
#         dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
#         dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
#
#         if (dx >= 0) and (dy >= 0):
#             return dx*dy
#         else:
#             return 0
#
# # each rectangle is [x,y,width,height]
# # x and y specifies one corner of the rectangle
# # width and height define the size in units along the x and y axes respectively.
# def overlap_ratio(rect1, rect2):
#     inter_area = rectint(rect1, rect2)
#     union_area = np.multiply(rect1[:, 2], rect1[:, 3]) + np.multiply(rect1[:, 2], rect1[:, 3]) - inter_area
#
#     r = np.divide(inter_area, union_area)
#     return r


# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def overlap_ratio(rect1, rect2):
    assert isinstance(rect1, (list, np.ndarray)) and isinstance(rect2, (list, np.ndarray))

    if len(np.array(rect1).shape) == 2 and len(np.array(rect2).shape) == 2:

        iou = []

        for _rect1, _rect2 in zip(rect1, rect2):

            boxA = [_rect1[0], _rect1[1], _rect1[0] + _rect1[2], _rect1[1] + _rect1[3]]
            boxB = [_rect2[0], _rect2[1], _rect2[0] + _rect2[2], _rect2[1] + _rect2[3]]

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            _iou = interArea / float(boxAArea + boxBArea - interArea)

            if _iou < 0:
                _iou = 0

            iou.append(_iou)
    else:
        assert len(np.array(rect1).shape) == len(np.array(rect2).shape)

        boxA = [rect1[0], rect1[1], rect1[0] + rect1[2], rect1[1] + rect1[3]]
        boxB = [rect2[0], rect2[1], rect2[0] + rect2[2], rect2[1] + rect2[3]]

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        if iou < 0:
            iou = 0

    # return the intersection over union value
    return iou
