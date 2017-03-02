import cv2
import numpy as np
from utils import mix_color_grad_thresh, adaptive_equalize_image, weighted_img, region_of_interest, draw_windows
from utils import CameraCalibrator, ProjectionManager, LineTracker
from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Read a new image
img = cv2.imread('./test_images/test16.jpg')


def process_image(frame):

    # Calibrate camera
    cam_calibration = CameraCalibrator(p_file='./camera_cal/calibration_mat.p', img_dir=None)
    mtx, dst, img_size = cam_calibration.get()

    row = img_size[0]
    col = img_size[1]
    # Un-distort image
    undst_img = adaptive_equalize_image(cv2.undistort(frame, mtx, dst), level=2)
    # Threshold image
    binary_img = mix_color_grad_thresh(undst_img, s_thresh=(180, 255), grad_thresh=(40, 90))

    # Perspective Transform
    projmgr = ProjectionManager(row, col, offset=300)
    binary_img = region_of_interest(binary_img, projmgr.get_roi())
    undst_birdeye = projmgr.get_birdeye_view(undst_img)
    birdeye_img = projmgr.get_birdeye_view(binary_img)

    # Lane Fitting and Tracking
    curve_centers = LineTracker(window_height=80, window_width=25, margin=15, ym=10/720, xm=4/384, smooth_factor=15)

    # Sliding window
    window_centroids = curve_centers.find_lane_line(warped=birdeye_img)
    result, leftx, rightx = draw_windows(birdeye_img, w=25, h=80, window_centroids=window_centroids)

    # Curve-fit
    lane_lines = curve_centers.curve_fit(result, leftx, rightx)

    # Convert back to normal view
    lane_lines = projmgr.get_normal_view(lane_lines)

    # Merge to original image
    lane_lines = cv2.addWeighted(undst_img, 1.0, lane_lines, 0.5, 0.0)
    return lane_lines

process_image(img)
# Plot the result
# f, ax = plt.subplots(2, 2)
# f.tight_layout()
# ax[0, 0].imshow(frame)
# ax[0, 0].set_title('Original Image')
# ax[0, 1].imshow(undst_img)
# ax[0, 1].set_title('Undistorted image')
# ax[1, 0].imshow(undst_birdeye, cmap='gray')
# ax[1, 0].set_title('Thresholded and ROI Cropped')
# ax[1, 1].imshow(lane_lines)
# ax[1, 1].set_title('Bird Eye View')
plt.show()

output = 'output.mp4'
clip1 = VideoFileClip("./project_video.mp4")
clip = clip1.fl_image(process_image)   # NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)
