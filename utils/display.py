import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2


def draw_box(image, box):
    im_with_bb = image.copy()
    cv2.rectangle(im_with_bb, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 0, 255))
    return im_with_bb


def display_result(image, box):
    im_with_bb = draw_box(image, box)
    cv2.imshow("result", im_with_bb)
    cv2.waitKey(1)

    return im_with_bb
