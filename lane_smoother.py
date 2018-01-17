import numpy as np
import cv2
from lane import FittedLane


# TODO remove lane smoother but only directly use a single FittedLane instance
class LaneSmoother(object):

    def __init__(self):
        self.lane = None
        pass

    def fit(self, img):
        """

        :param img: a binary image containing the lane pixels in a bird view
        :return: a FittedLane
        """
        if self.lane is None:
            self.lane = FittedLane.fit(img)
        else:
            self.lane.fit_adapt(img)

        return self.lane
