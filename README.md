## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![Video](https://raw.githubusercontent.com/dat-ai/advance-lane-finding/master/docs/gif.gif)](https://www.youtube.com/watch?v=blezjtz1lWU)


#### MAIN PIPELINE
------------------
```shell
def process_image(frame):
    global cam_calibration  # Camera Calibrator
    global img_filter       # Image Filter
    global projmgr          # Projection Manger - stored perspective transform matrix
    global curve_centers    # Find and Track Lane Line
    global debug            # Enable/Disable Debug Mode
    global diag_screen      # Diagnostic Screen

    mtx, dst, size = cam_calibration.get()
    # Un-distort image
    undst_img = cv2.undistort(frame, mtx, dst)
    # Threshold image
    bin_img = img_filter.mix_color_grad_thresh(undst_img, s_thresh=(88, 250),  h_thresh=(120, 250))
    # grad_thresh=(60, 130) can el
    # Perspective Transform
    binary_roi = img_filter.region_of_interest(bin_img, projmgr.get_roi())
    birdseye_view = projmgr.get_birdeye_view(undst_img)
    birdseye_img = projmgr.get_birdeye_view(binary_roi)
    # Sliding window
    window_centroids = curve_centers.find_lane_line(warped=birdseye_img)
    windows, left_x, right_x = draw_windows(birdseye_img, w=25, h=80, window_centroids=window_centroids)
    # Curve-fit and calculate curvature and offset
    curved_fit, curvature, offset = curve_centers.curve_fit(windows, left_x, right_x)
    # Convert back to normal view
    lane_lines = projmgr.get_normal_view(curved_fit)
    # Merge to original image
    lane_lines = cv2.addWeighted(undst_img, 1.0, lane_lines, 0.5, 0.0)
    # Add diagnostic screen if user needs to debug
    if debug is True:
        lane_lines = diag_screen.build(undst_img, lane_lines, bin_img, binary_roi,
                                       birdseye_img, birdseye_view, curved_fit, windows, curvature, offset)
    return lane_lines
```

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal/calibration_mat.p`. The images in `test_images` are for testing pipeline on single frames.  

If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

The video called `project_video.mp4` is the video your pipeline should work well on.  
The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  
The `harder_challenge.mp4` video is another optional challenge and is brutal!
