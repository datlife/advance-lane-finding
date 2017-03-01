import cv2
import numpy as np
from utils import mix_color_grad_thresh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread('test2.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Un-distort image

# Convert to lane-line
binary_img = mix_color_grad_thresh(img)

# Apply bird-eye view


# Lane fitting


# Lane Tracking
# Plot the result
plt.imshow(binary_img, cmap='gray')
plt.show()