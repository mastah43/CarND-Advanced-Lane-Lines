import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from camera_img_undistorter import CameraImageUndistorter
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane_isolator import LaneIsolator


def plot_images(img_src, img_dst):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_src)
    ax1.set_title('Source Image', fontsize=50)
    ax2.imshow(img_dst, cmap='gray')
    ax2.set_title('Destination Image', fontsize=50)
    ax2.imshow(img_dst, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)


def chessboard():
    img_transformer = CameraImagePerspectiveTransform()
    img_undistorter = CameraImageUndistorter()
    lane_isolator = LaneIsolator()

    img = cv2.cvtColor(cv2.imread('./test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)
    img_undistorted = img_undistorter.undistort(img)

    #plot_images(img, img_unwarped)

    img_binary = lane_isolator.isolate_lanes(img_undistorted)
    plot_images(img_undistorted, img_binary)

    # TODO use magnitude of gradient
    # TODO use direction of gradient
    # TODO filter interesting colors

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # TODO filter area where lanes can be

    img_birdview = img_transformer.to_birdview(img_binary)
    #plot_images(img_undistorted, img_birdview)

    # TODO detect lane lines
    histogram = np.sum(img_birdview[img_birdview.shape[0] // 2:, :], axis=0)
    #plt.plot(histogram)
    #plt.show(block=True)

    # TODO fit polynomial


if __name__ == '__main__':
    chessboard()

