import numpy as np
import cv2


def adaptive_equalize_image(img, level):
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


def hsv_image(img, yellow_min=(85, 40, 170), yellow_max=(99, 200, 255), white_min=(0, 0, 210), white_max=(140, 20, 255)):
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