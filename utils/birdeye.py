import cv2
import numpy as np


class ProjectionManager(object):
    def __init__(self, row, col, offset=100):
        self.col = col
        self.row = row
        self.offset = offset
        self.src = np.float32([[[col * 0.10, row],              # bottom-left
                           [col * 0.90, row],                   # bottom-right
                           [col * 0.53, row * 0.60],            # top-right
                           [col * 0.47, row * 0.60]]])          # top-left
        self.dst = np.float32([[col * 0.10 + offset, row],      # bottom left
                          [col * 0.90 - offset, row],           # bottom right
                          [col, 0],                             # top right
                          [0, 0]])                              # top left
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inversed = cv2.getPerspectiveTransform(self.dst, self.src)

    def get_birdeye_view(self, img):
        # Warp image to a top-down view
        warped = cv2.warpPerspective(img, self.M, (self.col, self.row), flags=cv2.INTER_LINEAR)
        return warped

    def get_normal_view(self, bird_eye_img):
        warped = cv2.warpPerspective(bird_eye_img, self.M_inversed, (self.col, self.row), flags=cv2.INTER_LINEAR)
        return warped