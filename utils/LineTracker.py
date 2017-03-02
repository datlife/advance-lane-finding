import cv2
import numpy as np


class LineTracker(object):
    def __init__(self):
        self.window_width = None
        self.window_height = None