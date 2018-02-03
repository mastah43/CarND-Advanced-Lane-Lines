import numpy as np
import cv2
import collections
from functools import reduce
import itertools
import matplotlib.pyplot as plt


class LaneSpec:
    min_width_meters = 3.7


class LaneImageSpec:
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = LaneSpec.min_width_meters / 640  # meters per pixel in x dimension
    width = 1280
    height = 720


class LaneLine(object):
    def __init__(self):
        self.fit_pix = None
        self.fit_meters = None

        frame_count_to_smooth_fits = 5
        self.fits_history_pix = collections.deque([], frame_count_to_smooth_fits)
        self.fits_history_meters = collections.deque([], frame_count_to_smooth_fits)
        self.fit_pix = None
        self.fit_meters = None
        self.fit_last_pix = None
        self.fit_last_meters = None
        self.fit_rejected_count_total = 0
        self.fits_rejected_in_a_row_count = 0
        self.detected = False

    @staticmethod
    def x_for_fit(fit, y_pixels):
        return fit[0] * y_pixels ** 2 + fit[1] * y_pixels + fit[2]

    def x_pixels(self, y_pixels):
        return LaneLine.x_for_fit(self.fit_pix, y_pixels)

    def x_meters(self, y_pixels):
        y_meters = y_pixels * LaneImageSpec.ym_per_pix
        return self.fit_meters[0] * y_meters ** 2 + self.fit_meters[1] * y_meters + self.fit_meters[2]

    def curve_radius_meters(self):
        y_eval = LaneImageSpec.height
        return ((1 + (2 * self.fit_meters[0] * y_eval * LaneImageSpec.ym_per_pix + self.fit_meters[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.fit_meters[0])

    @staticmethod
    def smooth_fit(fits):
        return np.average(fits, axis=0)

    @staticmethod
    def is_fit_outlier(fit, fit_last):
        y_pixels = np.linspace(0, LaneImageSpec.height - 1, 10)
        return reduce((lambda a, b: a or b),
                      map((lambda y: abs((LaneLine.x_for_fit(fit, y) / LaneLine.x_for_fit(fit_last, y)) - 1) > 0.05),
                          y_pixels))

    def invalidate_last_fit(self):
        self.fits_history_pix.pop()
        self.fits_history_meters.pop()
        self.detected = False
        if len(self.fits_history_pix) > 0:
            self.fit_pix = LaneLine.smooth_fit(self.fits_history_pix)
            self.fit_meters = LaneLine.smooth_fit(self.fits_history_meters)
        else:
            self.fit_pix = self.fit_last_pix
            self.fit_meters = self.fit_last_meters

    def fit(self, nonzerox, nonzeroy, x_base:int, trace_img=None):
        """
        Returns a fit polynome for the given potential lane pixels
        :param nonzerox:
        :param nonzeroy:
        :param x_base:
        :param trace_img:
        :return: fit coefficients for pixels, fit coefficients for meters
        """

        lane_pix_indices = self.find_lane_pixel_indices(nonzerox, nonzeroy, trace_img, x_base)

        # can not fit lane polynomial if too few pixels were found
        if np.sum(lane_pix_indices) < 20:
            self.fits_rejected_in_a_row_count += 1
            return None

        # Trace: colorize the lane pixels
        if trace_img is not None:
            trace_img[nonzeroy[lane_pix_indices], nonzerox[lane_pix_indices]] = [255, 0, 0]

        # Extract lane line pixel positions
        lane_pix_x = nonzerox[lane_pix_indices]
        lane_pix_y = nonzeroy[lane_pix_indices]

        # Fit second order polynomial for pixels and meters
        fit_new_pix = np.polyfit(lane_pix_y, lane_pix_x, 2)
        fit_new_meters = np.polyfit(lane_pix_y * LaneImageSpec.ym_per_pix, lane_pix_x * LaneImageSpec.xm_per_pix, 2)

        # Reject fit if too far away from last fit but recover after several frames
        fit_good = None
        if (len(self.fits_history_pix) == 0) or not LaneLine.is_fit_outlier(fit_new_pix, self.fit_pix):
            self.fits_rejected_in_a_row_count = 0
            self.fits_history_pix.append(fit_new_pix)
            self.fits_history_meters.append(fit_new_meters)
            self.fit_last_pix = self.fit_pix
            self.fit_last_meters = self.fit_meters
            self.fit_pix = LaneLine.smooth_fit(self.fits_history_pix)
            self.fit_meters = LaneLine.smooth_fit(self.fits_history_meters)
            self.detected = True
        else:
            self.fit_rejected_count_total += 1
            self.fits_rejected_in_a_row_count += 1
            self.fits_history_pix.popleft()
            self.fits_history_meters.popleft()
            self.detected = False
            if len(self.fits_history_pix) > 0:
                self.fit_pix = LaneLine.smooth_fit(self.fits_history_pix)
                self.fit_meters = LaneLine.smooth_fit(self.fits_history_meters)

        # Trace: Draw fit poly - green for good, yellow for bad
        if trace_img is not None:
            img_height = trace_img.shape[0]
            plot_y = np.linspace(0, img_height - 1, img_height)
            plot_x = LaneLine.x_for_fit(fit_new_pix, plot_y)
            plot_pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))])
            plot_color = (0, 255, 0) if self.detected else (255, 255, 0)
            cv2.polylines(trace_img, np.int32([plot_pts]), isClosed=False, color=plot_color, thickness=8)

    def find_lane_pixel_indices(self, nonzerox, nonzeroy, trace_img, x_base):

        lane_pix_indices = []
        # width of the windows +/- margin
        margin = 80

        if self.detected:
            nonzerox_fit_last = LaneLine.x_for_fit(self.fit_pix, nonzeroy)
            lane_pix_indices = (
                (nonzerox > (nonzerox_fit_last - margin)) &
                (nonzerox < (nonzerox_fit_last + margin)))
        else:
            # number of sliding windows
            window_count = 9

            # height of windows
            window_height = np.int(LaneImageSpec.height / window_count)

            # minimum number of pixels found to recenter window
            min_pix_in_window = 50

            # Step through the windows one by one
            x_current = x_base
            for window in range(window_count):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = LaneImageSpec.height - (window + 1) * window_height
                win_y_high = LaneImageSpec.height - window * window_height
                win_x_low = x_current - margin
                win_x_high = x_current + margin

                # Draw the windows on the visualization image
                if trace_img is not None:
                    cv2.rectangle(trace_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

                # Identify the nonzero pixels in x and y within the window
                window_pix_indices = (
                    (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                    (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                # TODO idea: remove outliers from lane pixels using dbscan clustering

                # Append these indices to the lists
                lane_pix_indices.append(window_pix_indices)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(window_pix_indices) > min_pix_in_window:
                    x_current = np.int(np.mean(nonzerox[window_pix_indices]))

            # Concatenate the arrays of indices
            lane_pix_indices = np.concatenate(lane_pix_indices)

        return lane_pix_indices


    @staticmethod
    def create_fit(nonzerox, nonzeroy, x_base, trace_img=None):
        line = LaneLine()
        line.fit(nonzerox, nonzeroy, x_base, trace_img)
        return line


class FittedLane(object):

    def __init__(self, line_left:LaneLine, line_right:LaneLine):
        self.line_left = line_left
        self.line_right = line_right
        self.lane_width_too_narrow_count = 0
        self.lane_lines_not_parallel_count = 0
        self.last_ok = None

    def deviation_from_lane_center_meters(self):
        y = LaneImageSpec.height - 1
        left_x = self.line_left.x_pixels(y)
        right_x = self.line_right.x_pixels(y)
        camera_x = (right_x - left_x) / 2 + left_x
        center_x = LaneImageSpec.width / 2
        deviation_x = camera_x - center_x
        deviation_meters = deviation_x * LaneImageSpec.xm_per_pix
        return deviation_meters

    def lane_radius_meters(self):
        left_curve_rad = self.line_left.curve_radius_meters()
        right_curve_rad = self.line_right.curve_radius_meters()
        return (left_curve_rad + right_curve_rad) / 2

    def lane_width_meters(self):
        y = LaneImageSpec.height - 1
        left_x_meters = self.line_left.x_meters(y)
        right_x_meters = self.line_right.x_meters(y)
        return abs(right_x_meters - left_x_meters)

    def _are_lines_near_parallel(self):
        radius_left = self.line_left.curve_radius_meters()
        radius_right = self.line_right.curve_radius_meters()
        return abs(float(radius_left)/float(radius_right) - 1) < 0.5

    def _lines_ok(self):
        lines_ok = True
        lane_width = self.lane_width_meters()
        if lane_width < LaneSpec.min_width_meters:
            # lane not properly detected if lane lines are too narrow
            self.lane_width_too_narrow_count += 1
            lines_ok = False

        if not self._are_lines_near_parallel():
            self.lane_lines_not_parallel_count += 1
            lines_ok = False

        return lines_ok

    def fit_adapt(self, img, trace_img=None):
        leftx_base, rightx_base = FittedLane._lines_left_right_x_bottom(img)
        nonzerox, nonzeroy = FittedLane._lane_pixels_xy(img)

        if trace_img is not None:
            trace_img[nonzeroy, nonzerox] = [255, 255, 255]

        fit_left = self.line_left.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=leftx_base, trace_img=trace_img)
        fit_right = self.line_right.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=rightx_base, trace_img=trace_img)

        # remove last fits if the two lane lines are not good together
        if not self._lines_ok():
            if self.line_left.detected:
                self.line_left.invalidate_last_fit()
            if self.line_right.detected:
                self.line_right.invalidate_last_fit()
            self.last_ok = False
        else:
            self.last_ok = True

    @staticmethod
    def _lines_left_right_x_bottom(img):
        """
        Find the peak of the left and right halves of the histogram.
        These will be the starting point for the left and right lines.
        :param img:
        :return:
        """

        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    @staticmethod
    def _lane_pixels_xy(img):
        """
        Identify the x and y positions of all nonzero pixels in the image
        :param img:
        :return:
        """
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        return nonzerox, nonzeroy

    @staticmethod
    def fit(img, trace_img=None):
        """
        :param img: a binary image containing mostly left and right lane markings
        :return: a FittedLane
        """

        leftx_base, rightx_base = FittedLane._lines_left_right_x_bottom(img)
        nonzerox, nonzeroy = FittedLane._lane_pixels_xy(img)

        line_left = LaneLine.create_fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=leftx_base, trace_img=trace_img)
        line_right = LaneLine.create_fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=rightx_base, trace_img=trace_img)

        return FittedLane(line_left, line_right)



