import cv2
import numpy as np

class LaneIsolator(object):

    def __init__(self,
                 ksize=3,
                 gradx_thresh=(20, 100),
                 grady_thresh=(20, 100),
                 mag_thresh=(20, 100),
                 dir_thresh = (0, np.pi / 8)):
        self.ksize = ksize
        self.gradx_thresh = gradx_thresh
        self.grady_thresh = grady_thresh
        self.mag_thresh = mag_thresh
        self.dir_thresh = dir_thresh
        self._sobelx = None
        self._sobely = None
        self._gray = None

    def _abs_sobel_thresh(self, img, sobel, thresh_min=0, thresh_max=255):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        abs_sobel = np.absolute(sobel)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)  # TODO why create a copy?
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary_output

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def _mag_thresh(self, img, sobelx, sobely, mag_thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function to threshold an image for a given range and Sobel kernel
    def _dir_threshold(self, img, sobelx, sobely, thresh=(0, np.pi / 2)):
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def _color_threshold(self, img, thresh = (90, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]

        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1
        return binary

    def isolate_lanes(self, img):
        self._gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY, dst=self._gray)
        self._sobelx = cv2.Sobel(src=self._gray, ddepth=cv2.CV_64F, dx=1, dy=0, dst=self._sobelx, ksize=self.ksize)
        self._sobely = cv2.Sobel(src=self._gray, ddepth=cv2.CV_64F, dx=0, dy=1, dst=self._sobely, ksize=self.ksize)

        """ TODO needed?
        gradx = self._abs_sobel_thresh(
            img=img, orient='x', sobel=self._sobelx,
            thresh_min=self.gradx_thresh[0], thresh_max=self.gradx_thresh[1])
        grady = self._abs_sobel_thresh(
            img=img, orient='y', sobel=self._sobely,
            thresh_min=self.grady_thresh[0], thresh_max=self.grady_thresh[1])
            """
        mag_binary = self._mag_thresh(img, sobelx=self._sobelx, sobely=self._sobely, mag_thresh=self.mag_thresh)
        dir_binary = self._dir_threshold(img, sobelx=self._sobelx, sobely=self._sobely, thresh=self.dir_thresh)
        color_binary = self._color_threshold(img, thresh = (90, 255))

        # TODO allow debug output of images

        lanes = np.zeros_like(dir_binary)
        lanes[(dir_binary == 1) & (mag_binary == 1) & (color_binary == 1)] = 1
        return lanes
