## Advanced Lane Finding Project

---

**by Marc Neumann**

The goals / steps of this project are the following:

TODO
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image-calibration-undistorted]: ./output_images/calibration1-undistorted.jpg "Calibration Undistorted"
[image-undistorted]: ./output_images/undistorted.jpg "Undistorted"
[image-lase-mask]: ./output_images/lane_mask.jpg
[image-transform-source]: ./output_images/straight_lines1_source.jpg
[image-transform-result]: ./output_images/straight_lines1_source.jpg
[image-lane-mask-birdview]: ./output_images/lane_mask_birdview.jpg
[image-lane-git]: ./output_images/lane_fit.jpg
[image-augmented]: ./output_images/augmented_image.jpg
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

The class LaneIsolator in lane_isolator.py implements producing a binary mask image containing the lane pixels.
I used only a filter for white and yellow color.
TODO

![alt text][image-lase-mask]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The class CameraImagePerspectiveTransform in camera_birdview_transform.py implements the perspective transformation.
I am using opencv's getPerspectiveTransform function to compute a transformation matrix. I use the source points 
depicted in the following image where the lane lines are straight.

![alt text][image-transform-source]

I verified that my perspective transform was working as expected by seeing that the birdview transformation result
of the straight line image shows parallel lane lines:

![alt text][image-transform-result]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
