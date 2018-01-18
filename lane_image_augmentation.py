import cv2
import numpy as np
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane import FittedLane


class LaneImageAugmenter(object):
    def __init__(self, img_transformer:CameraImagePerspectiveTransform):
        self.img_transformer:CameraImagePerspectiveTransform = img_transformer

    def draw_all(self, dst, lane:FittedLane):
        return self.draw_lanes(dst, lane)

    def draw_statistics(self, dst, lane:FittedLane):
        radius_str = "radius: {0:.0f} m".format(lane.lane_radius_meters())
        deviation_str = "deviation: {0:.3f}m".format(lane.deviation_from_lane_center_meters())
        lane_width_str = "width: {0:.2f}m".format(lane.lane_width_meters())
        lane_width_str = "width: {0:.2f}m".format(lane.lane_width_meters())
        left_not_detected_str = "left not detected: {0}".format(lane.left_not_detected_count)
        right_not_detected_str = "right not detected: {0}".format(lane.right_not_detected_count)
        width_too_narrow_str = "width too narrow: {0}".format(lane.lane_width_too_narrow_count)
        lines_not_parallel_str = "lines not parallel: {0}".format(lane.lane_lines_not_parallel_count)

        cv2.putText(dst, radius_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, deviation_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lane_width_str, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, left_not_detected_str, (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, right_not_detected_str, (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, width_too_narrow_str, (700, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lines_not_parallel_str, (700, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)

    def draw_lanes(self, dst, lane:FittedLane):
        # TODO option to specify destination image

        # Create an image to draw the lines on
        warp_zero = np.zeros((dst.shape[0], dst.shape[1])).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Generate points for plotting
        img_width = color_warp.shape[0]
        ploty = np.linspace(0, img_width- 1, img_width)
        left_fitx = lane.line_left.x_pixels(ploty)
        right_fitx = lane.line_right.x_pixels(ploty)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.img_transformer.birdview_to_camera(color_warp)

        # write text for radius and deviation
        self.draw_statistics(newwarp, lane)

        # Combine the result with the original image
        result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)
        return result