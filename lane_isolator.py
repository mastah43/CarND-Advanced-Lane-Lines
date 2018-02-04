import cv2
import numpy as np


class LaneIsolator(object):

    def __init__(self,
                 ksize=3,
                 gradx_thresh=(20, 100),
                 grady_thresh=(20, 100),
                 mag_thresh=(20, 100),
                 dir_thresh = (0, np.pi / 8),
                 trace_intermediate_image=(lambda label,img : None)):
        """

        :param ksize:
        :param gradx_thresh:
        :param grady_thresh:
        :param mag_thresh:
        :param dir_thresh:
        :param trace_intermediate_image: For receiving intermediary images in lane isolation.
            Expects function(label:str, image). The image given to this function by LaneIsolater
            could be change after the call (passed by reference). Copy it if e.g. you want to plot it later.
        """
        self.ksize = ksize
        self.gradx_thresh = gradx_thresh
        self.grady_thresh = grady_thresh
        self.mag_thresh = mag_thresh
        self.dir_thresh = dir_thresh
        self._sobelx = None
        self._sobely = None
        self._gray = None
        self.trace_intermediate_image = trace_intermediate_image

    def _abs_sobel_thresh(self, sobel, thresh_min=0, thresh_max=255):
        abs_sobel = np.absolute(sobel)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        sobel_mask = np.zeros_like(scaled_sobel)  # TODO do not create copy
        sobel_mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return sobel_mask

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def _mag_thresh(self, sobelx, sobely, mag_thresh=(0, 255)):
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        sobel_mag_mask = np.zeros_like(gradmag)
        sobel_mag_mask[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        return sobel_mag_mask

    # Define a function to threshold an image for a given range and Sobel kernel
    def _dir_threshold(self, sobelx, sobely, thresh=(-np.pi / 2, np.pi / 2)):
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        sobel_dir_mask = np.zeros_like(absgraddir) # TODO .astype(np.uint8
        sobel_dir_mask[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return sobel_dir_mask

    def _color_threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        yellow_hls_low = np.array([0, 100, 100])
        yellow_hls_high = np.array([255, 255, 255])

        white_rgb_low = np.array([200, 200, 200])
        white_rgb_high = np.array([255, 255, 255])

        yellow_mask = cv2.inRange(hls, yellow_hls_low, yellow_hls_high)
        self.trace_intermediate_image('yellow color mask', yellow_mask)

        white_mask = cv2.inRange(img, white_rgb_low, white_rgb_high)
        self.trace_intermediate_image('white color mask', white_mask)

        color_mask = cv2.bitwise_or(yellow_mask, white_mask)
        return color_mask

    def isolate_lanes(self, img):
        #self._gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY, dst=self._gray)
        #self.trace_intermediate_image('converted to gray scale', self._gray)

        #self._sobelx = cv2.Sobel(src=self._gray, ddepth=cv2.CV_64F, dx=1, dy=0, dst=self._sobelx, ksize=self.ksize)
        #self.trace_intermediate_image('sobel x', self._sobelx)

        #self._sobely = cv2.Sobel(src=self._gray, ddepth=cv2.CV_64F, dx=0, dy=1, dst=self._sobely, ksize=self.ksize)
        #self.trace_intermediate_image('sobel y', self._sobely)

        #gradx = self._abs_sobel_thresh(
        #    sobel=self._sobelx, thresh_min=self.gradx_thresh[0], thresh_max=self.gradx_thresh[1])
        #self.trace_intermediate_image('gradx', gradx)

        #grady = self._abs_sobel_thresh(
        #    sobel=self._sobely, thresh_min=self.grady_thresh[0], thresh_max=self.grady_thresh[1])
        #self.trace_intermediate_image('grady', grady)

        #mag_binary = self._mag_thresh(sobelx=self._sobelx, sobely=self._sobely, mag_thresh=self.mag_thresh)
        #self.trace_intermediate_image('magnitude', mag_binary)

        #dir_binary = self._dir_threshold(sobelx=self._sobelx, sobely=self._sobely, thresh=self.dir_thresh)
        #self.trace_intermediate_image('direction', dir_binary)

        color_binary = self._color_threshold(img)
        self.trace_intermediate_image('color mask', color_binary)

        #lanes = np.zeros_like(dir_binary)
        #lanes[((dir_binary == 1) & (mag_binary == 1) & (gradx == 1) & (grady == 1)) | (color_binary == 1)] = 1
        #lanes[(color_binary == 1)] = 1
        #self.trace_intermediate_image('lanes', lanes)

        return color_binary
