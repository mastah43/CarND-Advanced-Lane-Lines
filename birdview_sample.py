from camera_birdview_transform import CameraImagePerspectiveTransform
import cv2

if __name__ == '__main__':
    img_perspective_transformer = CameraImagePerspectiveTransform()
    img = cv2.imread("./output_images/straight_lines1_source.jpg")
    img_birdview = img_perspective_transformer.to_birdview(img)
    cv2.imwrite("./output_images/straight_lines1_transformed.jpg", img_birdview)

