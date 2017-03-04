import cv2
import numpy as np
from utils import  weighted_img, region_of_interest, draw_windows, mix_color_grad_thresh
from utils import CameraCalibrator, ProjectionManager, LineTracker, ImageFilter
from moviepy.editor import VideoFileClip


def diagnostic_screen(lane_lines, edge_img, birdeye_img, birdeye_view, curved_fit, windows):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    status_screen = np.zeros((120, 1280, 3), dtype=np.uint8)
    cv2.putText(status_screen, 'Estimated lane curvature: ERROR!', (30, 60), font, 1, (255, 0, 0), 2)
    cv2.putText(status_screen, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255, 0, 0), 2)

    # Convert image to RGB
    edge_img = np.dstack((edge_img, edge_img, edge_img))*255
    birdeye_img = np.dstack((birdeye_img, birdeye_img, birdeye_img))*255

    # Assemble Diagnostic Screen
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Main output image
    diagScreen[0:720, 0:1280] = lane_lines
    # Top right 1
    diagScreen[0:240, 1280:1600] = cv2.resize(birdeye_view, (320, 240), interpolation=cv2.INTER_AREA)
    # Top right 2
    diagScreen[0:240, 1600:1920] = cv2.resize(windows, (320, 240), interpolation=cv2.INTER_AREA)*4
    diagScreen[240:480, 1280:1600] = cv2.resize(birdeye_img, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(edge_img, (320, 240), interpolation=cv2.INTER_AREA)
    diagScreen[600:1080, 1280:1920] = cv2.resize(curved_fit, (640, 480), interpolation=cv2.INTER_AREA) * 4
    diagScreen[720:840, 0:1280] = status_screen
    # Histogram here
    # diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320, 240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320, 240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320, 240), interpolation=cv2.INTER_AREA)
    # diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320, 240), interpolation=cv2.INTER_AREA)
    return diagScreen


def process_image(frame):
    global cam_calibration
    global img_filter
    global projmgr
    global curve_centers
    global debug

    mtx, dst, size = cam_calibration.get()

    # Un-distort image
    undst_img = cv2.undistort(frame, mtx, dst)

    # Threshold image
    bin_img = img_filter.mix_color_grad_thresh(undst_img)

    # Perspective Transform
    binary_roi = region_of_interest(bin_img, projmgr.get_roi())
    birdeye_view = projmgr.get_birdeye_view(undst_img)
    birdeye_img = projmgr.get_birdeye_view(binary_roi)

    # Sliding window
    window_centroids = curve_centers.find_lane_line(warped=birdeye_img)
    windows, leftx, rightx = draw_windows(birdeye_img, w=25, h=80, window_centroids=window_centroids)

    # Curve-fit
    curved_fit = curve_centers.curve_fit(windows, leftx, rightx)

    # Convert back to normal view
    lane_lines = projmgr.get_normal_view(curved_fit)

    # Merge to original image
    lane_lines = cv2.addWeighted(undst_img, 1.0, lane_lines, 0.5, 0.0)

    # Add diagnostic screen if user needs to debug
    if debug is True:
        lane_lines = diagnostic_screen(lane_lines, binary_roi, birdeye_img, birdeye_view, curved_fit, windows)

    return lane_lines

if __name__ == "__main__":
    # Camera Calibrator
    cam_calibration = CameraCalibrator(p_file='./camera_cal/calibration_mat.p', img_dir=None)
    mtx, dst, img_size = cam_calibration.get()

    # Image filtering
    img_filter = ImageFilter(img_size)

    # Projection Manger
    projmgr = ProjectionManager(cam_calibration, img_size[0], img_size[1], offset=300)

    # Lane Tracker
    curve_centers = LineTracker(window_height=80, window_width=25, margin=15, ym=10 / 720, xm=4 / 384, smooth_factor=20)

    # Debug
    debug = True

    # Create output video
    output = 'output.mp4'
    clip1 = VideoFileClip("./project_video.mp4")
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)
