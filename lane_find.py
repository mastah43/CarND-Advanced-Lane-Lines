from moviepy.editor import VideoFileClip
from camera_img_undistorter import CameraImageUndistorter
from camera_birdview_transform import CameraImagePerspectiveTransform
from lane_isolator import LaneIsolator
from lane_smoother import LaneSmoother
from lane import FittedLane
from lane_image_augmentation import LaneImageAugmenter


def write_lane_augmentation_video(src_video_file:str, dst_video_file:str):
    img_transformer = CameraImagePerspectiveTransform()
    img_undistorter = CameraImageUndistorter()
    lane_isolator = LaneIsolator()
    lane_img_augmenter = LaneImageAugmenter(img_transformer)
    lane_smoother = LaneSmoother()

    clip = VideoFileClip(src_video_file)

    def process_image(img):
        img_undistorted = img_undistorter.undistort(img)
        img_binary = lane_isolator.isolate_lanes(img_undistorted)
        img_binary_birdview = img_transformer.to_birdview(img_binary)
        lane = lane_smoother.fit(img_binary_birdview)
        return lane_img_augmenter.draw_all(img_undistorted, lane)

    clip_cut = clip
    #clip_cut = clip.subclip(19, 24)
    clip_augmented = clip_cut.fl_image(process_image)
    clip_augmented.write_videofile(dst_video_file, audio=False, progress_bar=True)


if __name__ == '__main__':
    #write_lane_augmentation_video('project_video.mp4', 'project_video_result_critical.mp4')
    write_lane_augmentation_video('challenge_video.mp4', 'challenge_video_result.mp4')
    write_lane_augmentation_video('harder_challenge_video.mp4', 'harder_challenge_video_result.mp4')

