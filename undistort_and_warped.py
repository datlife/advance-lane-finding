import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
print(mtx, dist)
# Read in an image
img = cv2.imread('test_img.png')
nx = 8  # the number of inside corners in x
ny = 6  # the number of inside corners in y

# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE


def corners_unwarp(img, nx, ny, mtx, dist):

    # Pass in your image into this function
    # Write code to do the following steps
    pattern_size = (nx, ny)   # Size of chessboard
    # 1) Un-distort using mtx and dist
    undistorted_img = cv2.undistort(img, mtx, dist)
    # 2) Convert to gray-scale
    gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    found_corners, corners = cv2.findChessboardCorners(gray_img, pattern_size)
    # 4) If corners found:
    M = None
    warped = None
    if found_corners:
            # a) draw corners
            new_img = cv2.drawChessboardCorners(undistorted_img, pattern_size, corners, found_corners)

            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
            # One especially smart way to do this would be to use four well-chosen corners that were
            # automatically detected during the un-distortion steps
            # For source points I'm grabbing the outer four detected corners
            src = np.float32([corners[0], corners[nx - 1],
                              corners[-1], corners[-nx]])

            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            img_size = (gray_img.shape[1], gray_img.shape[0])
            offset = 100
            dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
                             [img_size[0] - offset, img_size[1] - offset],
                             [offset, img_size[1] - offset]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
            warped = cv2.warpPerspective(new_img, M, img_size)
    return warped, M

top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
