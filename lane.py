import numpy as np
import cv2


class LaneImageSpec:
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 640  # meters per pixel in x dimension
    width = 1280
    height = 720


class LaneLine(object):
    def __init__(self, fit_pix, fit_meters):
        self.fit_pix = fit_pix
        self.fit_meters = fit_meters

    def x_pixels(self, y_pixels):
        return self.fit_pix[0] * y_pixels ** 2 + self.fit_pix[1] * y_pixels + self.fit_pix[2]

    def curve_radius_meters(self):
        y_eval = LaneImageSpec.height
        return ((1 + (2 * self.fit_meters[0] * y_eval * LaneImageSpec.ym_per_pix + self.fit_meters[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.fit_meters[0])

    @staticmethod
    def fit(nonzerox, nonzeroy, x_base:int, out_img):

        img_height = out_img.shape[0]
        img_width = out_img.shape[1]
        # Choose the number of sliding windows
        nwindows = 9

        x_current = x_base
        # Set height of windows
        window_height = np.int(out_img.shape[0] / nwindows)
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_height - (window + 1) * window_height
            win_y_high = img_width - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            win_xright_low = x_current - margin
            win_xright_high = x_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            window_inds = (
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(window_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(window_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[window_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # can not fit lane polynome if too few pixels were found
        if (len(lane_inds) < 2):
            return None

        # Colorize the lane pixels
        # TODO
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Extract left and right line pixel positions
        leftx = nonzerox[lane_inds]
        lefty = nonzeroy[lane_inds]

        # Fit a second order polynomial
        fit_pix = np.polyfit(lefty, leftx, 2)
        fit_meters = np.polyfit(lefty * LaneImageSpec.ym_per_pix, leftx * LaneImageSpec.xm_per_pix, 2)

        return LaneLine(fit_pix=fit_pix, fit_meters=fit_meters)


class FittedLane(object):

    def __init__(self, line_left:LaneLine, line_right:LaneLine, out_img):
        self.line_left = line_left
        self.line_right = line_right
        self.out_img = out_img

    def deviation_from_lane_center_meters(self):
        y = self.out_img.shape[0] - 1
        left_x = self.line_left.x_pixels(y)
        right_x = self.line_right.x_pixels(y)
        camera_x = (right_x - left_x) / 2 + left_x
        center_x = self.out_img.shape[1] / 2
        deviation_x = camera_x - center_x
        deviation_meters = deviation_x * LaneImageSpec.xm_per_pix
        return deviation_meters

    def lane_radius_meters(self):
        left_curve_rad = self.line_left.curve_radius_meters()
        right_curve_rad = self.line_right.curve_radius_meters()
        return (left_curve_rad + right_curve_rad) / 2

    @staticmethod
    def fit(img):
        """
        :param img: a binary image containing mostly left and right lane markings
        :return: a FittedLane
        """

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        line_left = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=leftx_base, out_img=out_img)
        line_right = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=rightx_base, out_img=out_img)

        return FittedLane(line_left, line_right, out_img)



