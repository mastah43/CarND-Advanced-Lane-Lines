import numpy as np
import cv2
from lane import FittedLane


class LaneSmoother(object):

    def __init__(self):
        self.lane = None
        pass

    def fit(self, img, trace_img=None):
        """

        :param img: a binary image containing the lane pixels in a bird view
        :return: a FittedLane
        """
        if self.lane is None:
            self.lane = FittedLane.fit(img, trace_img)
        else:
            self.lane.fit_adapt(img, trace_img)

        return self.lane
