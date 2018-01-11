import numpy as np
import glob
import cv2
import logging

class CameraImageUndistorter(object):
    """
        Supports undistorting camera images after beeing calibrated with given chessboard
        images from the very same camera.
    """

    mtx = None
    dist = None
    nx = 9
    ny = 6

    def __init__(self):
        obj_points = []
        img_points = []
        img_shape = None

        obj_points_single = np.zeros((self.nx * self.ny, 3), np.float32)
        obj_points_single[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)  # TODO explain

        image_files = glob.glob('./camera_cal/calibration*.jpg')
        for img_file in image_files:
            img = cv2.imread(img_file)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img_shape is None:
                img_shape = gray.shape

            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            if ret:
                img_points.append(corners)
                obj_points.append(obj_points_single)
            else:
                logging.warning("camera calibration could not use the following image - not all chessboard corners were found: " + img_file)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[::-1], None, None)

    def undistort(self, img, img_out=None):
        """
        Undistorts the given image.
        :param img: input image
        :return: undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)