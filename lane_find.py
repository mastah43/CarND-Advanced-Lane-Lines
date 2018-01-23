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
    img_transformer = CameraImagePerspectiveTransform()
    img_undistorter = CameraImageUndistorter()
    lane_isolator = LaneIsolator()
    lane_img_augmenter = LaneImageAugmenter(img_transformer)
    lane_smoother = LaneSmoother()

    clip = VideoFileClip(src_video_file)

    def write_image(name, img):
        file_path = "./output_images/" + name + ".jpg"
        color_conversion = cv2.COLOR_GRAY2BGR if ((len(img.shape) == 2) or (img.shape[2] == 1)) else cv2.COLOR_RGB2BGR
        cv2.imwrite(file_path, cv2.cvtColor(src=img, code=color_conversion))
        logging.info("wrote processing intermediate image to " + file_path)

    frame_to_write = [10]

    def process_image(img):
        frame_to_write[0] -= 1
        write_images = False # frame_to_write[0] == 0

        img_undistorted = img_undistorter.undistort(img)
        if write_images:
            write_image('undistorted', img_undistorted)

        img_binary = lane_isolator.isolate_lanes(img_undistorted)
        if write_images:
            write_image('lane_mask', img_binary)

        img_binary_birdview = img_transformer.to_birdview(img_binary)
        if write_images:
            write_image('lane_mask_birdview', img_binary_birdview)

        fit_trace_img = np.zeros_like(img_undistorted) # TODO reuse an image and zero it
        lane = lane_smoother.fit(img_binary_birdview, fit_trace_img)
        if write_images:
            write_image('lane_fit', fit_trace_img)

        if write_images:
            write_image('augmented_image', lane_img_augmenter.draw_all(np.copy(img_undistorted), lane))

        return lane_img_augmenter.draw_all(img_undistorted, lane, [img_binary, fit_trace_img])

    clip_cut = clip
    #clip_cut = clip.subclip(19, 24)
    clip_augmented = clip_cut.fl_image(process_image)
    clip_augmented.write_videofile(dst_video_file, audio=False, progress_bar=True)


if __name__ == '__main__':
    write_lane_augmentation_video('project_video.mp4', 'project_video_result.mp4')
    write_lane_augmentation_video('challenge_video.mp4', 'challenge_video_result.mp4')
    write_lane_augmentation_video('harder_challenge_video.mp4', 'harder_challenge_video_result.mp4')

