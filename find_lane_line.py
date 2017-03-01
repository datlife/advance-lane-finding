import cv2
import numpy as np
from utils import mix_color_grad_thresh, CameraCalibration, bird_eye_view
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Calibrate camera
cam_calibration = CameraCalibration(p_file= 'calibration_mat.p', img_dir=None)
cam_calibration.export_pickle('calibration_mat.p')
mtx, dst, img_size = cam_calibration.get()

# Un-distort image
img = cv2.imread('./test_images/test3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undst_img = cv2.undistort(img, mtx, dst)

# Convert to lane-line
binary_img = mix_color_grad_thresh(undst_img, s_thresh=(130, 255))

# Apply bird-eye view
birdeye_img, M = bird_eye_view(undst_img)


# Lane fitting


# Lane Tracking
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