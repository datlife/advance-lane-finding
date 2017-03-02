import cv2
import numpy as np
from utils import mix_color_grad_thresh, CameraCalibration, ProjectionManager
from utils import adaptive_equalize_image, weighted_img, region_of_interest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Calibrate camera
cam_calibration = CameraCalibration(p_file='calibration_mat.p', img_dir=None)
mtx, dst, img_size = cam_calibration.get()
row = img_size[0]
col = img_size[1]

# Read a new image
img = cv2.imread('./test_images/test16.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Un-distort image
undst_img = adaptive_equalize_image(cv2.undistort(img, mtx, dst), level=2)

# Threshold image
binary_img = mix_color_grad_thresh(undst_img, s_thresh=(150, 255), grad_thresh=(30, 100))

# Perspective Transform
projmgr = ProjectionManager(row, col, offset=100)
birdeye_img = projmgr.get_birdeye_view(undst_img)

# Lane Fitting and Tracking


# Plot the result
f, ax = plt.subplots(2, 2)
f.tight_layout()
ax[0, 0].imshow(img)
ax[0, 0].set_title('Original Image')
ax[0, 1].imshow(undst_img)
ax[0, 1].set_title('Undistorted image')
ax[1, 0].imshow(binary_img, cmap='gray')
ax[1, 0].set_title('Thresholded Grad. Dir.')
ax[1, 1].imshow(birdeye_img, cmap='gray')
ax[1, 1].set_title('Bird Eye View')
plt.show()