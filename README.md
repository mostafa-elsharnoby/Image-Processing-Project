# Image-Processing-Project

## Overview

The following pipeline was performed for lane detection:

* Apply a perspective transform ("birds-eye view") on original image.
* Get region of interest for warped image.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply binary thresholding.
* Detect lane pixels using Sliding Window Algorithm and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Unwarp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Dependencies
* Python 3.5
* Numpy
* OpenCV-Python
* Matplotlib
* Sys

## How to run
Run `./script.sh ./project_video.mp4 ./output.mp4` . This will take the raw video file at 'project_video.mp4', and create output video file at 'output.mp4' 
Run `./script.sh ./project_video.mp4 ./output.mp4 --debugging_mode` . This will take the raw video file at 'project_video.mp4', and create output video file at 'output.mp4' with activation of debugging mode
To run the script on any other video , change './project_video.mp4' with the appropriate video file path

## Pipeline

The various steps involved in the pipeline are as follows, each of these also been discussed in more detail in the sub section below:
* Apply a perspective transform to rectify image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* 
### Perspective Transformation & ROI selection
At this step we wrap the image into a 'bird's eye view' scene. which make it easier to detect lane lines (since they are relative parrallel) and measure their curvature.
* Firstly, we compute the transformation matrix by passing the ```src``` and ```dst``` points into ```cv2.getPerspectiveTransform```. These points are determined empirically with the help of the suite of test images.
* Then, the undistorted image is warped by passing it into ```cv2.warpPerspective``` along with the transformation matrix
*  Finally, we cut/crop out the sides of the image using a utility function ```get_roi()``` since this portion of the image contains no relevant information

### Generating a thresholded binary image
This was by far the most involved and challenging step of the pipeline. after testing the images and videos we notice that.

* Lane lines have one of two colours, white or yellow
* The surface on *both* sides of the lane lines has different brightness and/or saturation and a different hue than the line itself, and,
* Lane lines are not necessarily contiguous, so the algorithm needs to be able to identify individual line segments as belonging to the same lane line.

The latter property is addressed in the next two subsections whereas this subsection leverages the former two properties to develop a *filtering process* that takes an undistorted warped image and generates a thresholded binary image that only highlights the pixels that are likely to be part of the lane lines. Moreover, the thresholding/masking process needs to be robust enough to account for **uneven road surfaces** and most importantly **non-uniform lighting conditions**.

Many techniques such as gradient thresholding, thresholding over individual colour channels of different color spaces and a combination of them were experimented with over a training set of images with the aim of best filtering the lane line pixels from other pixels. The experimentation yielded the following key insights:

1. The performance of indvidual color channels varied in detecting the two colors (white and yellow) with some transforms significantly outperforming the others in detecting one color but showcasing poor performance when employed for detecting the other. Out of all the channels of RGB, HLS, HSV and LAB color spaces that were experiemented with the below mentioned provided the greatest signal-to-noise ratio and robustness against varying lighting conditions:

    * *White pixel detection*: R-channel (RGB) and L-channel (HLS)
    * *Yellow pixel detection*: B-channel (LAB) and S-channel (HLS)
2. Owing to the uneven road surfaces and non-uniform lighting conditions a **strong** need for **Adaptive Thresholding** wasn't realised
3. Gradient thresholding didn't provide any performance improvements over the color thresholding methods employed above, and hence, it was not used in the pipeline.

The final solution used in the pipeline consisted of an **ensemble of threshold masks**. Some of the key callout points are:
* Five masks were used, namely, RGB, HLS, HSV, LAB and a custom adaptive mask
* Each of these masks were composed through a *Logical OR* of two sub-masks created to detect the two lane line colors of yellow and white. Moreover, the threshold values associated with each sub-mask was adaptive to the mean of image / search window.

* The *custom adaptive mask* used in the ensemble leveraged the OpenCV ```cv2.adaptiveThreshold``` API with a Gaussian kernel for computing the threshold value. The construction process for the mask was similar to that detailed above with one important mention to the constructuion of the submasks:
    * White submask was created through a Logical AND of RGB R-channel and HSV V-channel, and, 
    * Yellow submask was created through a Logical AND of LAB B-channel and HLS S-channel

### Lane Line detection: Sliding Window technique
We now have a *warped, thresholded binary image* where the pixels are either 0 or 1; 0 (black color) constitutes the unfiltered pixels and 1 (white color) represents the filtered pixels. The next step involves mapping out the lane lines and  determining explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

The first technique employed to do so is: **Peaks in Histogram & Sliding Windows**

1. We first take a histogram along all the columns in the lower half of the image. This involves adding up the pixel values along each column in the image. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. These are used as starting points for our search. 

2. From these starting points, we use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

The parameters used for the sliding window search are:
```
   nb_windows = 12 # number of sliding windows
   margin = 100 # width of the windows +/- margin
   minpix = 50 # min number of pixels needed to recenter the window
   min_lane_pts = 10  # min number of 'hot' pixels needed to fit a 2nd order polynomial as a lane line
```

## Draw colored lane area on original image
Given all the above, we can annotate the original image with the lane area, and information about the lane curvature and vehicle offset. Below are the steps to do so:

* Create a blank image, and draw our polyfit lines (estimated left and right lane lines)
* Fill the area between the lines (with green color)
* Use the inverse warp matrix calculated from the perspective transform, to "unwarp" the above such that it is aligned with the original image's perspective
* Overlay the above annotation on the original image

The code to perform the above is in the function `draw_colored_lane()` in 'Lane_Detection.py'.
