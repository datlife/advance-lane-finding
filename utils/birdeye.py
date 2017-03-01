import cv2
import numpy as np


def bird_eye_view(img, offset=100):
    row = img.shape[0]
    col = img.shape[1]

    src = np.float32([[row*0.3, 0],
                      [row*0.8, 0],
                      [row*0.55, col*0.65],
                      [row*0.45, col*0.65]])
    dst = np.float32([[row*0.3, 0],
                      [row*0.8, 0],
                      [row*0.8, col],
                      [row*0.3, col]])

    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (col, row), flags=cv2.INTER_LINEAR)
    return warped, M