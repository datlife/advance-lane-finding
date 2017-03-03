import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Define a function that applies Sobel x or y,
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2, 3) Take the absolute derivative in x or y given orient = 'x' or 'y'
    abs_sobel = None
    if orient is 'x':
        abs_sobel = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0))
    if orient is 'y':
        abs_sobel = np.abs(cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    if abs_sobel is not None:
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # masked everything to dark ( = 0)
        abs_sobel_output = np.zeros_like(scaled_sobel)
        # if any pixel has thresh_min < value < thresh_max
        abs_sobel_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return abs_sobel_output
    else:
        return None


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradient_magnitude)/255
    gradient_magnitude = np.uint8(gradient_magnitude/scale_factor)

    # 5) Create a binary mask where mag thresholds are met
    mag_binary_output = np.zeros_like(gradient_magnitude)
    mag_binary_output[(gradient_magnitude >= mag_thresh[0]) & (gradient_magnitude <= mag_thresh[1])] = 1
    return mag_binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Calculate the direction of the gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    dir_binary_output = np.zeros_like(grad_direction)
    # 6) Return this mask as your binary_output image
    dir_binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
    return dir_binary_output


# # Read in an image and grayscale it
# image = mpimg.imread('signs_vehicles_xygrad.jpg')
#
# # Run the function
# grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(grad_binary, cmap='gray')
# ax2.set_title('Thresholded Gradient', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
#
# # Run the function
# mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(mag_binary, cmap='gray')
# ax2.set_title('Thresholded Magnitude', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# # Run the function
# dir_binary = dir_threshold(image, sobel_kernel=9, thresh=(0.7, 1.3))  # 0.7 --> 1.3 is line angle around 45* -> -45*
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(dir_binary, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# # Run the function
# gradx = abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
# grady = abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
# mag_binary = mag_thresh(image, mag_thresh=(30,100), sobel_kernel=15)
# dir_binary = dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=15)
#
# combined = np.zeros_like(dir_binary)
# combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
#
# # Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(combined, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()