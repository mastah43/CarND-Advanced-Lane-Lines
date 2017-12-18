import numpy as np
import glob
import cv2
import logging

class CameraImagePerspectiveTransform(object):
    """
        Supports transforming camera images to birdview images.
    """
    M = None

    def __init__(self):
        img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])

        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
        self.M = cv2.getPerspectiveTransform(src, dst)


    def to_birdview(self, img):
        """
        Unwarps the given image
        :param img:
        :return:
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M, img_size)
