import numpy as np
import cv2

class LaneSpec:
    min_width_meters = 3.7


class LaneImageSpec:
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = LaneSpec.min_width_meters / 640  # meters per pixel in x dimension
    width = 1280
    height = 720


class LaneLine(object):
    def __init__(self, fit_pix, fit_meters):
        self.fit_pix = fit_pix
        self.fit_meters = fit_meters

    def x_pixels(self, y_pixels):
        return self.fit_pix[0] * y_pixels ** 2 + self.fit_pix[1] * y_pixels + self.fit_pix[2]

    def x_meters(self, y_pixels):
        y_meters = y_pixels * LaneImageSpec.ym_per_pix
        return self.fit_meters[0] * y_meters ** 2 + self.fit_meters[1] * y_meters + self.fit_meters[2]

    def curve_radius_meters(self):
        y_eval = LaneImageSpec.height
        return ((1 + (2 * self.fit_meters[0] * y_eval * LaneImageSpec.ym_per_pix + self.fit_meters[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.fit_meters[0])

    @staticmethod
    def fit(nonzerox, nonzeroy, x_base:int, out_img=None):

        # Choose the number of sliding windows
        nwindows = 9

        x_current = x_base
        # Set height of windows
        window_height = np.int(LaneImageSpec.height / nwindows)
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = LaneImageSpec.height - (window + 1) * window_height
            win_y_high = LaneImageSpec.width - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            win_xright_low = x_current - margin
            win_xright_high = x_current + margin

            # Draw the windows on the visualization image
            if out_img is not None:
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

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.left_not_detected_count = 0
        self.right_not_detected_count = 0
        self.lane_width_too_narrow_count = 0
        self.lane_lines_not_parallel_count = 0

    def deviation_from_lane_center_meters(self):
        y = LaneImageSpec.height - 1
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
    def lane_radius_meters2(line_left : LaneLine, line_right : LaneLine):
        left_curve_rad = line_left.curve_radius_meters()
        right_curve_rad = line_right.curve_radius_meters()
        return (left_curve_rad + right_curve_rad) / 2

    @staticmethod
    def lane_width_meters2(line_left : LaneLine, line_right : LaneLine):
        y = LaneImageSpec.height - 1
        left_x_meters = line_left.x_meters(y)
        right_x_meters = line_right.x_meters(y)
        return abs(right_x_meters - left_x_meters)

    def lane_width_meters(self):
        return FittedLane.lane_width_meters2(self.line_left, self.line_right)

    @staticmethod
    def _are_lines_near_parallel(line_left : LaneLine, line_right : LaneLine):
        radius_left = line_left.curve_radius_meters()
        radius_right = line_right.curve_radius_meters()
        return abs(radius_left - radius_right) / abs(radius_left) < 0.5

    def _sanity_check_lines(self, line_left, line_right):
        line_detected = True
        if line_left is None:
            self.left_not_detected_count += 1
            line_detected = False
        if line_right is None:
            self.right_not_detected_count += 1
            line_detected = False
        if not line_detected:
            return

        lines_ok = True
        lane_width = FittedLane.lane_width_meters2(line_left, line_right)
        if lane_width < LaneSpec.min_width_meters:
            # lane not properly detected if lane lines are too narrow
            self.lane_width_too_narrow_count += 1
            lines_ok = False

        if FittedLane._are_lines_near_parallel(line_left, line_right):
            self.lane_lines_not_parallel_count += 1
            lines_ok = False

        # TODO check that slope of line is consistent (e.g. only one curve in it)

        return lines_ok

    def fit_adapt(self, img):
        leftx_base, rightx_base = FittedLane._lines_left_right_x_bottom(img)
        nonzerox, nonzeroy = FittedLane._lane_pixels_xy(img)
        line_left = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=leftx_base)
        line_right = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=rightx_base)
        line_left.curve_radius_meters()

        if not self._sanity_check_lines(line_left, line_right):
            return

        self.line_left = line_left
        self.line_right = line_right

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
    def fit(img):
        """
        :param img: a binary image containing mostly left and right lane markings
        :return: a FittedLane
        """

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255

        leftx_base, rightx_base = FittedLane._lines_left_right_x_bottom(img)
        nonzerox, nonzeroy = FittedLane._lane_pixels_xy(img)
        line_left = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=leftx_base, out_img=out_img)
        line_right = LaneLine.fit(nonzerox=nonzerox, nonzeroy=nonzeroy, x_base=rightx_base, out_img=out_img)

        return FittedLane(line_left, line_right, out_img)



