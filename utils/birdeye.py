import cv2
import numpy as np


class ProjectionManager(object):
    def __init__(self, row, col, offset=100):
        self.col = col
        self.row = row
        self.offset = offset
        self.src = np.float32([[[col * 0.05, row],              # bottom-left
                           [col * 0.95, row],                   # bottom-right
                           [col * 0.55, row * 0.62],            # top-right
                           [col * 0.45, row * 0.62]]])          # top-left
        self.dst = np.float32([[col*0.15 + offset, row],      # bottom left
                          [col*0.90 - offset, row],           # bottom right
                          [col*0.88-offset, 0],                             # top right
                          [col*0.10 + offset, 0]])                              # top left
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inversed = cv2.getPerspectiveTransform(self.dst, self.src)

    def get_birdeye_view(self, img):
        # Warp image to a top-down view
        warped = cv2.warpPerspective(img, self.M, (self.col, self.row), flags=cv2.INTER_LINEAR)
        return warped

    def get_normal_view(self, bird_eye_img):
        warped = cv2.warpPerspective(bird_eye_img, self.M_inversed, (self.col, self.row), flags=cv2.INTER_LINEAR)
        return warped

    def get_roi(self):
        return self.src.astype(int)