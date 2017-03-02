import cv2
import numpy as np
from utils.mask_roi import weighted_img

class LineTracker(object):
    def __init__(self, window_width, window_height, margin, xm, ym, smooth_factor):
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.ym_per_pixel = ym
        self.xm_per_pixel = xm
        self.smooth_factor = smooth_factor
        self.recent_centers = []           # Store recent windows (left,right)

    def find_lane_line(self, warped):
        wd_w = self.window_width
        wd_h = self.window_height
        margin = self.margin
        r = warped.shape[0]
        c = warped.shape[1]
        wd_centroids = []   # store (left, right) windows centroid position per level
        windows = np.ones(wd_w)  # Window Template for convolution

        # First find the two staring positions for the left and right lane using np.sum
        # to get the vertical image slice.

        # Then, use np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*r/4):, :int(c/2)], axis=0)
        l_center = np.argmax(np.convolve(windows, l_sum)) - wd_w/2
        r_sum = np.sum(warped[int(3*r/4):, int(c/2):], axis=0)
        r_center = np.argmax(np.convolve(windows, r_sum)) - wd_w/2 + int(c/2)

        # Add what we found for the first layer
        wd_centroids.append((l_center, r_center))

        # Iterate through each layer looking for max pixel locations
        for level in range(1, int(r/wd_h)):
            # convolve the window into the vertical slice of the image
            img_layer  = np.sum(warped[int(r - (level+1)*wd_h):int(r-level*wd_h), :], axis=0)
            conv_signal = np.convolve(windows, img_layer)
            # Find the est left centroid using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at the right side of window,
            # not the center of the window
            offset = wd_w/2
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, c))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # Find the  best right centroid
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, c))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            # Update window_centroids
            wd_centroids.append((l_center, r_center))

        self.recent_centers.append(wd_centroids)

        # Smooth the line
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)


def window_mask(width, height, img, center, level):
    row = img.shape[0]
    col = img.shape[1]
    output = np.zeros_like(img)
    output[int(row - (level+1)*height):int(row - level*height), max(0, int(center-width)):min(int(center+width), col)] = 1
    return output


def draw_windows(img, w, h, window_centroids):
    left_pts = np.zeros_like(img)
    right_pts = np.zeros_like(img)

    for level in range(0, len(window_centroids)):
        # Draw window
        l_mask = window_mask(w, h, img, window_centroids[level][0], level)
        r_mask = window_mask(w, h, img, window_centroids[level][1], level)
        # Add graphic points to window mask
        left_pts[(left_pts == 255) | (l_mask == 1)] = 255
        right_pts[(right_pts == 255) | (r_mask == 1)] = 255

    # Draw result
    template = np.array(left_pts+right_pts, dtype='uint8')
    zero_channel = np.zeros_like(template)
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), dtype='uint8')
    warpage = np.array(cv2.merge((img, img, img)), dtype='uint8')
    result = weighted_img(warpage, template,α=1.0, β=0.3, λ=0.)

    return result
