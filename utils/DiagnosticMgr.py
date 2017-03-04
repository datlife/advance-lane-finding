import cv2
import numpy as np
# Create better font
from PIL import ImageFont, ImageDraw, Image


class DiagnosticMgr(object):
    def __init__(self, img_filters, projection_mgr):
        self.filters = img_filters
        self.projection = projection_mgr
        self.font = "./docs/helvetica.ttf"

    def build(self, undst_img, lane_lines, bin_img,  edge_img, bird_eye_img, bird_eye_view, curved_fit, windows, curv, offset):

        # Assemble Diagnostic Screen
        diag_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Main output image
        diag_screen[0:720, 0:1280] = lane_lines
        diag_screen = self._build_top_right(diag_screen, curved_fit, bird_eye_view, bird_eye_img, windows)
        diag_screen = self._build_bottom_right(diag_screen, undst_img, bin_img)
        diag_screen = self._build_status(diag_screen, curv, offset)

        return diag_screen

    def _build_top_right(self, diag_screen,  curved_fit, bird_eye_view, bird_eye_img, windows):

        histogram = self.cal_lane_prob(bird_eye_img)
        # Convert image to RGB
        bird_eye_img = np.dstack((bird_eye_img, bird_eye_img, bird_eye_img)) * 255
        histogram = cv2.cvtColor(histogram, cv2.COLOR_GRAY2RGB)*255

        bird_eye_view = self._build_title(bird_eye_view, "Bird Eye View", size=40, h_offset=bird_eye_img.shape[0]*0.95)
        histogram = self._build_title(histogram, "Lane Line Probability", size=12, w_offset=2, h_offset=230)
        # Top right 1
        diag_screen[0:240, 1280:1600] = cv2.resize(bird_eye_view, (320, 240), interpolation=cv2.INTER_AREA)
        # Top right 2
        diag_screen[0:240, 1600:1920] = cv2.resize(windows, (320, 240), interpolation=cv2.INTER_AREA) * 4
        diag_screen[240:480, 1280:1600] = cv2.resize(histogram, (320, 240), interpolation=cv2.INTER_AREA)
        diag_screen[240:480, 1600:1920] = cv2.resize(curved_fit, (320, 240), interpolation=cv2.INTER_AREA)*4
        return diag_screen

    def _build_bottom_right(self, diag_screen, lane_lines, bin_img):

        # Build debug images
        abs_sobel_grad = self.filters.abs_sobel_thresh(lane_lines, thresh_min=30, thresh_max=100)
        mag_sobel_grad = self.filters.mag_thresh(lane_lines, mag_thresh=(30, 100))
        s_thresh = self.filters.hls_select(lane_lines, thresh=(88, 250))
        # Convert to RGB
        abs_sobel_grad = np.dstack((abs_sobel_grad, abs_sobel_grad, abs_sobel_grad)) * 255
        mag_sobel_grad = np.dstack((mag_sobel_grad, mag_sobel_grad, mag_sobel_grad)) * 255
        s_thresh = np.dstack((s_thresh, s_thresh, s_thresh)) * 255
        bin_img = np.dstack((bin_img, bin_img, bin_img))*255

        # Resize to fix small frame
        abs_sobel_grad = cv2.resize(abs_sobel_grad, (320, 240), interpolation=cv2.INTER_AREA)
        mag_sobel_grad = cv2.resize(mag_sobel_grad, (320, 240), interpolation=cv2.INTER_AREA)
        s_thresh = cv2.resize(s_thresh, (320, 240), interpolation=cv2.INTER_AREA)
        bin_img = cv2.resize(bin_img, (320, 240), interpolation=cv2.INTER_AREA)

        # Add Titles
        abs_sobel_grad = self._build_title(abs_sobel_grad, "Abs. Sobel Grad.", size=12, w_offset=2)
        mag_sobel_grad = self._build_title(mag_sobel_grad, "Magnitude Sobel Grad.", size=12, w_offset=2)
        s_thresh = self._build_title(s_thresh, "S Channel Thresh.", size=12, w_offset=2,color=(0, 0, 0, 255))
        bin_img = self._build_title(bin_img, "Combined Threshold", size=12, w_offset=2, color=(0, 0, 0, 255))
        # Display
        diag_screen[600:840, 1280:1600] = abs_sobel_grad
        diag_screen[600:840, 1600:1920] = mag_sobel_grad
        diag_screen[840:1080, 1280:1600] = s_thresh
        diag_screen[840:1080, 1600:1920] = bin_img
        return diag_screen

    def _build_status(self, diag_screen, curv, offset):
        # Drawing text in diagnostic pipeline.
        status_screen = np.zeros((120, 1280, 3), dtype=np.uint8)

        if offset < 0.55:
            offset_str = str(round(offset, 3)) + " m"
        else:
            offset_str = str(round(offset, 3)) + " m " + "(Warning: Offset is too high!)"
        add1 = self._build_title(status_screen, "Estimated Center Offset: " + offset_str, size=30, w_offset=30, h_offset=30)
        status_screen = self._build_title(add1, "Estimated Radius of Curvature: " + str(round(curv, 3)) + " km", size=30, w_offset=30, h_offset=70)
        diag_screen[720:840, 0:1280] = status_screen  # Show curvature, offset
        return diag_screen

    def _build_title(self, img, text, size=20, w_offset=20, h_offset=20, color=(255, 255, 255, 255)):
        im = Image.fromarray(img, 'RGB')
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype(font=self.font, size=size)
        position = (w_offset, h_offset)
        draw.text(position, text, font=font, fill=color)
        img = np.array(im)
        return img

    def cal_lane_prob(self, bird_eye_img, size=(320, 240)):
        sample = cv2.resize(bird_eye_img, size, interpolation=cv2.INTER_AREA)
        histogram = np.zeros_like(sample)
        lane_probs = np.array(np.sum(sample[sample.shape[0] / 2:, :], axis=0), dtype='int32') + 50
        ploty = np.linspace(0, sample.shape[1] - 1, sample.shape[1])

        x = np.concatenate((ploty, ploty[::-1]), axis=0)
        y = np.concatenate((lane_probs - 3, lane_probs[::-1] + 3), axis=0)
        data = np.array(list(zip(x, y)), dtype='int32')
        cv2.fillPoly(histogram, [data], color=[255, 255, 0])
        histogram = cv2.flip(histogram, 0)
        histogram = np.uint8(255 * histogram)
        return histogram

