import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread("test4.jpg")


def mix_color_grad_thresh(img, grad_thresh=(20, 90), s_thresh=(170, 255), dir_thresh=(0.7, 1.3), sobel_size=9):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_size))  # Take the derivative in x
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_size))  # Take the derivative in x
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold magnitude gradient
    gradient_magnitude = np.sqrt(abs_sobelx ** 2 + abs_sobely ** 2)
    scale_factor = np.max(gradient_magnitude) / 255
    gradient_magnitude = np.uint8(gradient_magnitude / scale_factor)
    # 5) Create a binary mask where mag thresholds are met
    mag_binary_output = np.zeros_like(gradient_magnitude)
    mag_binary_output[(gradient_magnitude >= grad_thresh[0]) & (gradient_magnitude <= grad_thresh[1])] = 1

    # Threshold direction gradient
    grad_direction = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary_output = np.zeros_like(grad_direction)
    dir_binary_output[(grad_direction >= dir_thresh[0]) & (grad_direction <= dir_thresh[1])] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1  # OR mean combine

    return combined_binary
#
# combined_binary = mix_color_grad_thresh(img, grad_thresh=(40, 100), s_thresh=(90, 255), dir_thresh=(0.8, 1.4))
# # Plotting thresholded images
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.set_title('Stacked thresholds')
# ax1.imshow(img)
#
# ax2.set_title('Combined S channel and gradient thresholds')
# ax2.imshow(combined_binary, cmap='gray')
# plt.show()