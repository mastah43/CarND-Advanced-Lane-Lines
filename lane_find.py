from moviepy.editor import VideoFileClip
from camera_img_undistorter import CameraImageUndistorter
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane_isolator import LaneIsolator
from lane_smoother import LaneSmoother
from lane import FittedLane
from lane_image_augmentation import LaneImageAugmenter
import numpy as np
import cv2
import logging


def write_lane_augmentation_video(src_video_file:str, dst_video_file:str):
    write_image_step = [1]
    write_frame = [-1] # set to positive value if a certain frames processing images should be written do disk

    def log_output_image(name, img):
        if frame_count[0] == write_frame:
            file_path = "./output_images/" + '{:02d}'.format(write_image_step[0]) + "_" + name.replace(" ", "_") + ".jpg"
            if img.dtype == np.float64:
                img = img.astype(np.uint8) # does not work with cv2.imwrite
            print("writing processing intermediate image to " + file_path)
            color_conversion = cv2.COLOR_GRAY2BGR if ((len(img.shape) == 2) or (img.shape[2] == 1)) else cv2.COLOR_RGB2BGR
            cv2.imwrite(file_path, cv2.cvtColor(src=img, code=color_conversion))
            write_image_step[0] += 1

    def trace_image_lane_isolation(name, img):
        log_output_image(name, img)

    img_transformer = CameraImagePerspectiveTransform()
    img_undistorter = CameraImageUndistorter()
    lane_isolator = LaneIsolator(
        ksize=15,
        gradx_thresh=(30, 255),
        grady_thresh=(10, 255),
        mag_thresh=(10, 255),
        dir_thresh=(-0.7, 0.7),
        trace_intermediate_image=trace_image_lane_isolation)
    lane_img_augmenter = LaneImageAugmenter(img_transformer)
    lane_smoother = LaneSmoother()

    clip = VideoFileClip(src_video_file)

    frame_count = [0]

    def process_image(img):

        img_undistorted = img_undistorter.undistort(img)
        log_output_image('undistorted', img_undistorted)

        img_binary = lane_isolator.isolate_lanes(img_undistorted)

        img_birdview = img_transformer.to_birdview(img_binary)
        log_output_image('birdview', img_birdview)

        fit_trace_img = np.zeros_like(img_undistorted) # TODO reuse an image and zero it
        lane = lane_smoother.fit(img_birdview, fit_trace_img)
        log_output_image('lane_fit', fit_trace_img)

        if frame_count[0] == write_frame and lane is not None:
            log_output_image(
                'augmented_image',
                lane_img_augmenter.draw_all(dst=np.copy(img_undistorted), lane=lane))

        frame_count[0] += 1
        if lane is not None:
            return lane_img_augmenter.draw_all(
                dst=img_undistorted, lane=lane,
                imgs_steps=[img_binary, fit_trace_img], frame_no=frame_count[0])
        else:
            return img

    clip_cut = clip
    #clip_cut = clip.subclip(19, 24) # TODO
    clip_augmented = clip_cut.fl_image(process_image)
    clip_augmented.write_videofile(dst_video_file, audio=False, progress_bar=True, ffmpeg_params=['-force_key_frames', 'expr:gte(t,n_forced*1)'])


if __name__ == '__main__':
    write_lane_augmentation_video('project_video.mp4', 'project_video_result.mp4')
    #write_lane_augmentation_video('challenge_video.mp4', 'challenge_video_result.mp4')
    #write_lane_augmentation_video('harder_challenge_video.mp4', 'harder_challenge_video_result.mp4')

