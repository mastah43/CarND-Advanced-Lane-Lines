import cv2
import numpy as np
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane import FittedLane


class LaneImageAugmenter(object):
    def __init__(self, img_transformer:CameraImagePerspectiveTransform):
        self.img_transformer:CameraImagePerspectiveTransform = img_transformer

    def draw_all(self, dst, lane:FittedLane, imgs_steps=None):
        return self.draw_lanes(dst, lane, imgs_steps)

    def draw_images_steps(self, img_result, imgs_steps):
        # TODO use function
        pass

    def draw_statistics(self, dst, lane:FittedLane):
        radius_str = "radius: {0:.0f} m".format(lane.lane_radius_meters())
        deviation_str = "deviation: {0:.3f}m".format(lane.deviation_from_lane_center_meters())
        lane_width_str = "width: {0:.2f}m".format(lane.lane_width_meters())
        lane_width_str = "width: {0:.2f}m".format(lane.lane_width_meters())
        left_not_detected_str = "left not detected: {0}".format(lane.line_left.fit_outlier_count)
        right_not_detected_str = "right not detected: {0}".format(lane.line_right.fit_outlier_count)
        width_too_narrow_str = "width too narrow: {0}".format(lane.lane_width_too_narrow_count)
        lines_not_parallel_str = "lines not parallel: {0}".format(lane.lane_lines_not_parallel_count)

        cv2.putText(dst, radius_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, deviation_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lane_width_str, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, left_not_detected_str, (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, right_not_detected_str, (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, width_too_narrow_str, (700, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)
        cv2.putText(dst, lines_not_parallel_str, (700, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color=255, thickness=4)

    def draw_lanes(self, dst, lane:FittedLane, imgs_steps=None):
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
        img_merge = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)

        if imgs_steps is not None:
            img_full_height = img_merge.shape[0]
            img_full_width = img_merge.shape[1]
            img_step_height = int(img_full_height / len(imgs_steps))
            img_step_width = 256
            img_lanes_width = img_full_width - img_step_width
            result = np.zeros((img_full_height, img_full_width, 3), dtype=np.uint8)
            result[0:img_full_height, 0:img_full_width, :] = cv2.resize(img_merge, (img_full_width, img_full_height))

            # show mask for lane pixels
            # TODO
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