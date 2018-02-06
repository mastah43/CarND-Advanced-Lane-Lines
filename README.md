## Advanced Lane Finding Project

---

**by Marc Neumann**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Prerequisites
In order to run the pipeline to produce the project result video at least python 3.6 is needed 
(since e.g. Python type hints are used in the code - see also https://www.python.org/dev/peps/pep-0484/).
It will not run with python 2.
You could also use the file [requirements.txt](requirements.txt) to setup a conda environment with all
the required dependencies but having the following python 3 versions of the packages
numpy, opencv, matplotlib installed should be sufficient.

## Files
The following files are contained:
- [result video](project_video_result.mp4)
- [jupyter notebook](lane_finding.ipynb)
- [python program to produce result video](lane_find.py)
- [camera calibration](camera_img_undistorter.py)
- [lane pixel identification](lane_isolator.py)
- [lane fitting](lane.py)
- [perspective tranformation](camera_birdview_transform.py)
- [lane drawing on undistorted original images](lane_image_augmentation.py)
- [the writeup](README.md)

[//]: # (Image References)
[image-calibration-undistorted]: ./output_images/calibration1-undistorted.jpg "Calibration Undistorted"

[image-transform-source]: ./output_images/straight_lines1_source.jpg
[image-transform-result]: ./output_images/straight_lines1_transformed.jpg
[image-undistorted]: ./output_images/01_undistorted.jpg "Undistorted"
[image-yellow-color-mask]: ./output_images/02_yellow_color_mask.jpg
[image-white-color-mask]: ./output_images/03_white_color_mask.jpg
[image-color-mask]: ./output_images/04_color_mask.jpg
[image-birdview]: ./output_images/05_birdview.jpg
[image-lane-fit]: ./output_images/06_lane_fit.jpg
[image-augmented]: ./output_images/07_augmented_image.jpg
[video-result]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

Camera calibration is implemented in class CameraImageUndistorter in file camera_img_undistorter.py. 
CameraImageUndistorter creates the gray scale versions of the calibration camera images of the same chessboard 
to get the camera matrix and distortion coefficients using opencv upon initialization. 
Then CameraImageUndistorter is used to undistort camera images using the previous determined camera matrix 
and distortion coefficients.
 
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In constructor of CameraImageUndistorter I start by preparing "object points", which will be the (x, y, z) coordinates 
of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, 
such that the object points are the same for each calibration image.  Thus, `obj_points` is just a replicated 
array of coordinates, and `obj_points_single` will be appended with a copy of it every time I successfully detect all 
chessboard corners in a calibration image.  The found chessboard corners as (x, y) pixel positions in each calibration 
image are appended to `img_points`.  

I then used the output `obj_points` and `img_points` to compute the camera calibration 
and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to a camera calibration image and obtained this result which clearly shows 
that the distortion is removed: 

![alt text][image-calibration-undistorted]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

A sample result for the distortion correction on a frame from the project video: 
![alt text][image-undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The class LaneIsolator in lane_isolator.py implements producing a binary mask image containing the potential lane pixels.
I experimented with a combination of color mask, x and y gradient, direction and magnitude sobel.
Finally I chose to used only a combined filter for white in RGB color space and yellow in LSV color space.
I am using LSV for yellow to be more robust against different lighting conditions.

Yellow color mask for the distorted image shown above:
![alt text][image-yellow-color-mask]

White color mask for the distorted image shown above:
![alt text][image-white-color-mask]

Combined yellow and white color mask:
![alt text][image-color-mask]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The class CameraImagePerspectiveTransform in camera_birdview_transform.py implements the perspective transformation.
I am using opencv's getPerspectiveTransform function to compute a transformation matrix. I use the source points 
depicted in the following image where the lane lines are straight.

![alt text][image-transform-source]

I verified that my perspective transform was working as expected by seeing that the birdview transformation result
of the straight line image shows parallel lane lines:

![alt text][image-transform-result]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The classes FittedLane and LaneLine in lane.py implement finding lane pixels and fitting a second order polynomial.
In function LaneLine.create_fit(...) a sliding window approach is implemented to determine the lane pixels.
The x position for the base of a lane line are determined using a histogram on the potential lane pixels.
The x position of the highest left peak is used as base x position for the left lane. 
Analogue the right peak for the right lane line. 
In a window with pixel width 160 and height of image height divided by 9 (number of windows) all potential lane pixels 
are considered lane pixels. If enough lane pixels are found the new center x position of the next window is realigned
using the mean x position of the found lane pixels.
Once all lane pixels have been found, the second order polynomial is fit to them.

![alt text][image-lane-fit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature radius is computed by FittedLane.lane_radius_meters() in file lane.py. 
It takes the average of the curvature radius of left and right lane line via LaneLine.curve_radius_meters().
The curvature radius calculation of a lane line uses the second order polynomial fit in meters.
I implemented the formula explained here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).
I am using the pixel to meters factors for the birdview image from class LaneImageSpec in lane.py.

The deviation from lane center is computed by FittedLane.deviation_from_lane_center_meters().
The x pixel position of left and lane line is determined using the fit polynomial of the lane lines.
Then the center of the lane is determine as the middle between left and right lane line.
The pixel deviation is derived using pixel lane center x and image width divided by 2 for the camera center x position.
The pixel deviation is then transformed to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Implementation of result plotting is done in class LaneImageAugmenter in file lane_image_augmentation.py.
I also include the potential lane pixels in the upper right corner and the windows used for finding lane pixels 
in the lower right corner.
Example for a frame in the video:

![alt text][image-augmented]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  

For the video pipeline I am smoothing the coefficients of the lane line polynomial (see LaneLine.add_fit)
using a history of the last fits. If a fit polynomial from the last frame exists I use this lane line polynomial
to find the isolated lane pixels in the current frame.
Also I am rejecting a fit polynomial if x positions are more then 5% away from last polynomial 
(see LaneLine.is_fit_outlier). I also reject new fits if the two lane lines do not have a reasonable distance
that matched the minimum width of a lane.

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  
Using white and yellow color masks solely for identifying lane line pixels worked since the lighting
conditions in the project video are good (sunny).

The pipeline will fail if either side of the lane boundary can not be detected still after more than 5 frames.
A problem for the pipeline could be yellow and white vehicles or other objects in front of the ego vehicle.
The white or yellow pixels could be falsely identified as lane line pixels and included into fitting the
polynomial for the lane lines. Another problem could be very bright road sections due to the sun or bright 
white reflections in the wind shield. Another problem are sections strong shadows on the lane lines.
 
To improve my pipeline I would include sobel magnitude and direction in lane pixel detection in order to be
more robust against detecting other white and yellow objects.  
