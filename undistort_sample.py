from camera_img_undistorter import CameraImageUndistorter
import cv2

if __name__ == '__main__':
    img_undistorter = CameraImageUndistorter()
    img = cv2.imread('./camera_cal/calibration1.jpg')
    img_undistorted = img_undistorter.undistort(img)
    cv2.imwrite("./output_images/calibration1-undistorted.jpg", img_undistorted)

