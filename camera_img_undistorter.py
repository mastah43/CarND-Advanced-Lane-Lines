import numpy as np
import glob
import cv2
import logging
import pickle
import os.path

class CameraImageUndistorter(object):
    """
        Supports undistorting camera images after beeing calibrated with given chessboard
        images from the very same camera.
    """
    def __init__(self):

        pickle_file_name = "undistort.p"

        if os.path.isfile(pickle_file_name):
            undistort_params = pickle.load(open(pickle_file_name, "rb"))
            self.mtx = undistort_params["mtx"]
            self.dist = undistort_params["dist"]
        else:
            obj_points = []
            img_points = []
            img_shape = None
            nx = 9
            ny = 6

            obj_points_single = np.zeros((nx * ny, 3), np.float32)
            obj_points_single[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

            image_files = glob.glob('./camera_cal/calibration*.jpg')
            for img_file in image_files:
                img = cv2.imread(img_file)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if img_shape is None:
                    img_shape = gray.shape

                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

                if ret:
                    img_points.append(corners)
                    obj_points.append(obj_points_single)
                else:
                    logging.warning("camera calibration could not use the following image - not all chessboard corners were found: " + img_file)

            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[::-1], None, None)

            undistort_params = {}
            undistort_params["mtx"] = self.mtx
            undistort_params["dist"] = self.dist
            pickle.dump(undistort_params, open(pickle_file_name, "wb"))

    def undistort(self, img):
        """
        Undistorts the given image.
        :param img: input image
        :return: undistorted image
        """
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)