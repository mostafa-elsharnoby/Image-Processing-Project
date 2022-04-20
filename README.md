# Image-Processing-Project

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

## Draw colored lane area on original image
Given all the above, we can annotate the original image with the lane area, and information about the lane curvature and vehicle offset. Below are the steps to do so:

* Create a blank image, and draw our polyfit lines (estimated left and right lane lines)
* Fill the area between the lines (with green color)
* Use the inverse warp matrix calculated from the perspective transform, to "unwarp" the above such that it is aligned with the original image's perspective
* Overlay the above annotation on the original image

The code to perform the above is in the function `draw_colored_lane()` in 'Lane_Detection.py'.
