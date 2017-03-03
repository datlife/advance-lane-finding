import cv2
import numpy as np


class ImageFilter(object):
    '''
    Handle image filtering
    '''

    def __init__(self, img_size):
        self.row = img_size[0]
        self.col = img_size[1]

    def abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
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
            scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
            # 5) Create a mask of 1's where the scaled gradient magnitude
            # masked everything to dark ( = 0)
            abs_sobel_output = np.zeros_like(scaled_sobel)
            # if any pixel has thresh_min < value < thresh_max
            abs_sobel_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
            return abs_sobel_output
        else:
            return None

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scale_factor = np.max(gradient_magnitude) / 255
        gradient_magnitude = np.uint8(gradient_magnitude / scale_factor)

        # 5) Create a binary mask where mag thresholds are met
        mag_binary_output = np.zeros_like(gradient_magnitude)
        mag_binary_output[(gradient_magnitude >= mag_thresh[0]) & (gradient_magnitude <= mag_thresh[1])] = 1
        return mag_binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
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

    def hls_select(self, img, thresh=(0, 255), channel=2):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # 2) Apply a threshold to the S channel
        s_channel = hls[:, :, channel]  # default is s_channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > thresh[0]) & (s_channel < thresh[1])] = 1
        # 3) Return a binary image of threshold result
        binary_output = s_binary  # placeholder line
        return binary_output

    def adaptive_equalize_image(self, img, level):
        """
        Equalize an image - Increase contrast for the image
            # http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

        :param img:    an gray image
        :param level:  clipLevel
        :return: a equalized image
        """
        # Conver BGR to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=level)
        cl = clahe.apply(l)
        merge = cv2.merge((cl, a, b))
        result = cv2.cvtColor(merge, cv2.COLOR_LAB2BGR)
        return result

    def hsv_image(self, img, yellow_min=(85, 40, 170), yellow_max=(99, 200, 255), white_min=(0, 0, 210),
                  white_max=(140, 20, 255)):
        """
        Convert BGR to HSV
        green = np.uint8([[[255,236,107]]])
        hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
        print(hsv_green)
        """
        img = adaptive_equalize_image(img, 3.0)

        # Convert to HSV image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        WHITE_MIN = np.array(white_min)
        WHITE_MAX = np.array(white_max)
        YELLOW_MIN = np.array(yellow_min)
        YELLOW_MAX = np.array(yellow_max)

        white_mask = cv2.inRange(hsv, WHITE_MIN, WHITE_MAX)
        yellow_mask = cv2.inRange(hsv, YELLOW_MIN, YELLOW_MAX)
        mask = cv2.addWeighted(white_mask, 1.0, yellow_mask, 1.0, 0.0)
        hsv = cv2.bitwise_and(img, img, mask=mask)
        return hsv

    def mix_threshold(self, image):
        # Sobel Threshold
        gradx = self.abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
        grady = self.abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
        mag_binary = self.mag_thresh(image, mag_thresh=(30, 100), sobel_kernel=15)
        dir_binary = self.dir_threshold(image, thresh=(0.7, 1.3), sobel_kernel=15)

        # Color Threshold
        s_binary = self.hls_select(image, thresh=(88, 250))
        h_binary = self.hls_select(image, thresh=(120, 250), channel=1)
        binimg = np.zeros_like(s_binary)
        binimg[(s_binary == 1) & (h_binary == 1)] = 1

        # Mix Threshold together
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (binimg == 1)] = 1

        return combined