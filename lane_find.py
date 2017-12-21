import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from camera_img_undistorter import CameraImageUndistorter
from camera_birdview_transform import CameraImagePerspectiveTransform

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # TODO only sobel x required?
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel) # TODO why create a copy?
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def plot_images(img_src, img_dst):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_src)
    ax1.set_title('Source Image', fontsize=50)
    ax2.imshow(img_dst, cmap='gray')
    ax2.set_title('Destination Image', fontsize=50)
    ax2.imshow(img_dst, cmap='gray') # TODO make a parameter
    #ax2.imshow(img_dst, cmap='binary') # TODO make a parameter
    #ax2.imshow(img_dst, cmap=ListedColormap(['w', 'k'])) # TODO make a parameter
    # TODO ax2.imshow(img_dst, cmap=map, interpolation='nearest') # TODO make a parameter
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def chessboard():
    img_transformer = CameraImagePerspectiveTransform()
    img_undistorter = CameraImageUndistorter()

    img = cv2.imread('./test_images/test1.jpg')
    img_undistorted = img_undistorter.undistort(img)
    #plot_images(img, img_unwarped)

    img_undistorted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_binary = abs_sobel_thresh(img_undistorted, thresh_min=20, thresh_max=100)
    #plot_images(img_undistorted, img_binary)

    # TODO use magnitude of gradient
    # TODO use direction of gradient
    # TODO filter interesting colors

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # TODO filter area where lanes can be

    img_birdview = img_transformer.to_birdview(img_binary)
    #plot_images(img_undistorted, img_birdview)

    # TODO create binary image of lane lines
    # TODO detect lane lines
    #img_sobel = abs_sobel_thresh(img)

    histogram = np.sum(img_birdview[img_birdview.shape[0] // 2:, :], axis=0)
    plt.plot(histogram)
    plt.show()


if __name__ == '__main__':
    chessboard()

