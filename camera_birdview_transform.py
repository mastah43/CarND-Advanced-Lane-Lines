import numpy as np
import glob
import cv2
import logging

class CameraImagePerspectiveTransform(object):
    """
        Supports transforming camera images to birdview images.
    """

    def __init__(self):
        img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])

        # Source points were taken from test image straight lines 1.
        # The test image was undistorted and then the left lane marking is
        # used as indicator for source points. Instead of using the right lane marking
        # the mirrored left lane markings are used in order to a symmetric perspective
        # transformation. See ./examples/straight_lines1_source_points.jpg
        src = np.float32([
            [598, 449],
            [208, img_size[1]],
            [img_size[0] - 208, img_size[1]],
            [img_size[0] - 598, 449]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])
        self.m = cv2.getPerspectiveTransform(src, dst)
        ret, self.m_inv = cv2.invert(self.m)

    def to_birdview(self, img):
        """
        Unwarps the given camera image to bird view.
        :param img:
        :return:
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.m, img_size)

    def birdview_to_camera(self, img):
        """
        Unwarps the given camera image to bird view.
        :param img:
        :return:
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.m_inv, img_size)