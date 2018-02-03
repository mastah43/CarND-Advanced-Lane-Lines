import cv2
import numpy as np
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane import FittedLane


class LaneImageAugmenter(object):
    def __init__(self, img_transformer:CameraImagePerspectiveTransform):
        self.img_transformer:CameraImagePerspectiveTransform = img_transformer

    def draw_all(self, dst, lane: FittedLane, imgs_steps=None, frame_no: int=None):
        img_augmented = self.draw_lane(dst, lane, imgs_steps)
        self.draw_statistics(img_augmented, lane, frame_no)
        return img_augmented

    def draw_statistics(self, dst, lane:FittedLane, frame_no : int=None):
        radius_str = "radius: {0:.0f} m".format(lane.lane_radius_meters())
        deviation_str = "deviation: {0:.3f}m".format(lane.deviation_from_lane_center_meters())
        lane_width_str = "width: {0:.2f}m".format(lane.lane_width_meters())
        left_not_detected_str = "left rejected in row: {0}".format(lane.line_left.fits_rejected_in_a_row_count)
        right_not_detected_str = "right rejected in row: {0}".format(lane.line_right.fits_rejected_in_a_row_count)
        width_too_narrow_str = "width too narrow: {0}".format(lane.lane_width_too_narrow_count)
        lines_not_parallel_str = "lines not parallel: {0}".format(lane.lane_lines_not_parallel_count)

        x_col1 = 10
        x_col2 = 450
        cv2.putText(dst, radius_str, (x_col1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, deviation_str, (x_col1, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lane_width_str, (x_col1, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, left_not_detected_str, (x_col2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, right_not_detected_str, (x_col2, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, width_too_narrow_str, (x_col2, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lines_not_parallel_str, (x_col2, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        if frame_no is not None:
            frame_str = "frame: {0}".format(frame_no)
            cv2.putText(dst, frame_str, (x_col2, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)

    def draw_lane(self, dst, lane:FittedLane, imgs_steps=None):
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
        color_green = (0, 255, 0)
        color_red = (255, 0, 0)
        cv2.fillPoly(color_warp, np.int_([pts]), color_green if lane.last_ok else color_red)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        img_lanes_camera = self.img_transformer.birdview_to_camera(color_warp)

        # Combine the result with the original image
        img_merge = cv2.addWeighted(dst, 1, img_lanes_camera, 0.3, 0)

        # draw step images
        if imgs_steps is not None:
            img_full_height = img_merge.shape[0]
            img_full_width = img_merge.shape[1]
            img_step_height = int(img_full_height / len(imgs_steps))
            img_step_width = 256
            img_lanes_width = img_full_width - img_step_width
            result = np.zeros((img_full_height, img_full_width, 3), dtype=np.uint8)
            result[0:img_full_height, 0:img_lanes_width, :] = cv2.resize(img_merge, (img_lanes_width, img_full_height))

            y_top_img_step = 0
            for img_step in imgs_steps:
                if (len(img_step.shape) == 2) or (img_step.shape[2] == 1):
                    result[y_top_img_step:y_top_img_step+img_step_height, img_lanes_width:img_lanes_width+img_step_width, 0] = \
                        cv2.resize(img_step, (img_step_width, img_step_height))
                else:
                    result[y_top_img_step:y_top_img_step + img_step_height,
                        img_lanes_width:img_lanes_width + img_step_width] = \
                        cv2.resize(img_step, (img_step_width, img_step_height))
                y_top_img_step += img_step_height
        else:
            result = img_merge

        return result