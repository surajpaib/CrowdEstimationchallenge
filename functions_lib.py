import cv2
import numpy as np

class CrowdCounter(object):
    def __init__(self):
        self.params = None

    def mutlifile_read(self, *args):
        for arg in args:
             yield cv2.imread(arg)

    def multifile_write(self, *args):
        for arg in args:
            cv2.imwrite(arg[0], arg[1])
        return

    def background_subtraction(self, img1, img2, img3):
        diff1 = cv2.absdiff(img1, img2)
        diff2 = cv2.absdiff(img2, img3)
        diff3 = cv2.absdiff(img3, img1)

        return diff1, diff2, diff3

    def high_pass_filtering(self, img):
            try:
                im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            except:
                im = img

            f = np.fft.fft2(im)
            fshift = np.fft.fftshift(f)
            rows, cols = im.shape
            crow, ccol = rows / 2, cols / 2
            fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)

            return img_back

    def morph_operations(self, kernel, im, operation):
        if operation == "open":
            return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        if operation == "close":
            return cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
        if operation == "erode":
            return cv2.morphologyEx(im, cv2.MORPH_ERODE, kernel)
        if operation == "dilate":
            return cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

    def blob_detect_set_params(self):
        params = cv2.SimpleBlobDetector_Params()
        # Detect circles
        params.filterByCircularity = True
        params.minCircularity = 0.15
        params.maxCircularity = 0.8
        # Threshold for splitting images
        params.minThreshold = 10
        params.maxThreshold = 500
        # filter by color
        params.filterByColor = False
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        # Filter by Inertia
        params.filterByArea = True
        params.minArea = 100

        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 0.5

        self.params = params
        return

    def run_blob_detector(self, processed_im, original_im):
        blob = cv2.SimpleBlobDetector(self.params)
        keypoint = blob.detect(processed_im)
        im_with_keypoints = cv2.drawKeypoints(original_im, keypoint, np.array([]), (255, 0, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints, keypoint
