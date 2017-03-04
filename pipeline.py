import cv2
from utils import  region_of_interest, draw_windows
from utils import CameraCalibrator, ProjectionManager, LineTracker, ImageFilter, DiagnosticMgr
from moviepy.editor import VideoFileClip


def process_image(frame):
    global cam_calibration
    global img_filter
    global projmgr
    global curve_centers
    global debug
    global diag_screen

    mtx, dst, size = cam_calibration.get()
    # Un-distort image
    undst_img = cv2.undistort(frame, mtx, dst)

    # Threshold image
    bin_img = img_filter.mix_color_grad_thresh(undst_img, s_thresh=(88, 250),  h_thresh=(120, 250))

    # Perspective Transform
    binary_roi = region_of_interest(bin_img, projmgr.get_roi())
    birdeye_view = projmgr.get_birdeye_view(undst_img)
    birdeye_img = projmgr.get_birdeye_view(binary_roi)

    # Sliding window
    window_centroids = curve_centers.find_lane_line(warped=birdeye_img)
    windows, leftx, rightx = draw_windows(birdeye_img, w=25, h=80, window_centroids=window_centroids)

    # Curve-fit
    curved_fit, curvature, offset = curve_centers.curve_fit(windows, leftx, rightx)

    # Convert back to normal view
    lane_lines = projmgr.get_normal_view(curved_fit)

    # Merge to original image
    lane_lines = cv2.addWeighted(undst_img, 1.0, lane_lines, 0.5, 0.0)

    # Add diagnostic screen if user needs to debug
    if debug is True:
        lane_lines = diag_screen.build(undst_img, lane_lines, bin_img, binary_roi,
                                       birdeye_img, birdeye_view, curved_fit, windows, curvature, offset)

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
    curve_centers = LineTracker(window_height=80, window_width=25, margin=30, ym=10 / 720, xm=4 / 384, smooth_factor=10)

    # Debug
    debug = True

    diag_screen = DiagnosticMgr(img_filter, projmgr)
    # Create output video
    output = 'output.mp4'
    clip1 = VideoFileClip("./challenge_video.mp4")
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(output, audio=False)
