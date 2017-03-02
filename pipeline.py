import cv2
import numpy as np
from utils import mix_color_grad_thresh, adaptive_equalize_image, weighted_img, region_of_interest, draw_windows
from utils import CameraCalibration, ProjectionManager, LineTracker
import matplotlib.pyplot as plt

# Read a new image
img = cv2.imread('./test_images/test16.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process_frame(img):
    # Calibrate camera
    cam_calibration = CameraCalibration(p_file='calibration_mat.p', img_dir=None)
    mtx, dst, img_size = cam_calibration.get()
    row = img_size[0]
    col = img_size[1]

    # Un-distort image
    undst_img = adaptive_equalize_image(cv2.undistort(img, mtx, dst), level=2)
    # Threshold image
    binary_img = mix_color_grad_thresh(undst_img, s_thresh=(180, 255), grad_thresh=(40, 90))

    # Perspective Transform
    projmgr = ProjectionManager(row, col, offset=300)
    binary_img = region_of_interest(binary_img, projmgr.get_roi())
    birdeye_img = projmgr.get_birdeye_view(binary_img)

    # Lane Fitting and Tracking
    curve_centers = LineTracker(window_height=80, window_width=25, margin=10, ym=10/720, xm=4/384, smooth_factor=15)
    window_centroids = curve_centers.find_lane_line(warped=birdeye_img)
    lane_lines = draw_windows(birdeye_img, w=25, h=80, window_centroids=window_centroids)

    # Convert back to normal view
    # Plot the result
    f, ax = plt.subplots(2, 2)
    f.tight_layout()
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original Image')
    ax[0, 1].imshow(undst_img)
    ax[0, 1].set_title('Undistorted image')
    ax[1, 0].imshow(birdeye_img, cmap='gray')
    ax[1, 0].set_title('Thresholded and ROI Cropped')
    ax[1, 1].imshow(window_img, cmap='gray')
    ax[1, 1].set_title('Bird Eye View')

process_frame(img)
plt.show()