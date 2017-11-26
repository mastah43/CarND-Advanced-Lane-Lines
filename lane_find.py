import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def chessboard():
    nx = 9
    ny = 5

    # Make a list of calibration images
    fname = 'camera_cal/calibration1.jpg'
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()
        print("ok")
    else:
        print("failed")

if __name__ == '__main__':
    chessboard()

