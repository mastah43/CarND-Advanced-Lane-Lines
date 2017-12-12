import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_img_undistorter import CameraImageUndistorter


def chessboard():
    img_undistorter = CameraImageUndistorter()
    img_test_undistorted = img_undistorter.undistort(cv2.imread('./camera_cal/calibration1.jpg'))
    plt.imshow(img_test_undistorted)
    plt.show()

if __name__ == '__main__':
    chessboard()

