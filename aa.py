import cv2

import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片

import sys
import numpy as np
import glob


class StereoCalibration(object):
    def __init__(self):
        self.imagesL = self.read_images('imgL')
        self.imagesR = self.read_images('imgR')

    def read_images(self, cal_path):
        filepath = glob.glob(cal_path + '/*.jpg')
        filepath.sort()
        return filepath


    def calibration_photo(self):

        x_nums = 12
        y_nums = 6

        world_point = np.zeros((x_nums * y_nums, 3), np.float32)
        world_point[:, :2] = np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)

        world_position = []
        image_positionl = []
        image_positionr = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

        for ii in range(1, 9):

            image_path_l = self.imagesL[ii]
            image_path_r = self.imagesR[ii]

            image_l = cv2.imread(image_path_l)
            image_r = cv2.imread(image_path_r)
            gray_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2GRAY)
            gray_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)


            #         ok,corners = cv2.findChessboardCorners(gray,(x_nums,y_nums),None)
            #             ok1,cornersl = cv2.findChessboardCorners(gray_l,(x_nums,y_nums),None)
            #             ok2,cornersr = cv2.findChessboardCorners(gray_r,(x_nums,y_nums),None)
            ok1, cornersl = cv2.findChessboardCorners(gray_l, (x_nums, y_nums), None)
            ok2, cornersr = cv2.findChessboardCorners(gray_r, (x_nums, y_nums), None)

            self.world = world_point
            #print(ok1 & ok2)
            if ok1 & ok2:

                world_position.append(world_point)

                exact_cornersl = cv2.cornerSubPix(gray_l, cornersl, (25, 13), (-1, -1), criteria)
                exact_cornersr = cv2.cornerSubPix(gray_r, cornersr, (25, 13), (-1, -1), criteria)

                image_positionl.append(exact_cornersl)
                image_positionr.append(exact_cornersr)

        #             image = cv2.drawChessboardCorners(image,(x_nums,y_nums),exact_corners,ok)
        #             cv2.imshow('image_corner',image)
        #             cv2.waitKey(0)

        image_shape = gray_l.shape[::-1]

        retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(world_position, image_positionl, image_shape, None,
                                                                None)
        retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(world_position, image_positionr, image_shape, None,
                                                                None)
        print('ml = ', mtxl)
        print('mr = ', mtxr)
        print('dl = ', distl)
        print('dr = ', distr)
        stereo.m1 = mtxl
        stereo.m2 = mtxr
        stereo.d1 = distl
        stereo.d2 = distr


        self.cal_error(world_position, image_positionl, mtxl, distl, rvecsl, tvecsl)
        self.cal_error(world_position, image_positionr, mtxr, distr, rvecsr, tvecsr)


        self.stereo_calibrate(world_position, image_positionl, image_positionr, mtxl, distl, mtxr, distr, image_shape)

    def cal_error(self, world_position, image_position, mtx, dist, rvecs, tvecs):
        # 计算偏差
        mean_error = 0
        for i in range(len(world_position)):
            image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(image_position[i], image_position2, cv2.NORM_L2) / len(image_position2)
            mean_error += error
        print("total error: ", mean_error / len(image_position))

    def stereo_calibrate(self, objpoints, imgpoints_l, imgpoints_r, M1, d1, M2, d2, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l,
            imgpoints_r, M1, d1, M2,
            d2, dims,
            criteria=stereocalib_criteria, flags=flags)
        print(R)
        print(T)
        stereo.R = R
        stereo.T = T

class shuangmu:
    def __init__(self):
        self.m1 = [[1.41450411e+03, 0.00000000e+00, 3.29504889e+02], [0.00000000e+00, 1.42333006e+03, 2.31387604e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        self.m2 = [[2.22184933e+03, 0.00000000e+00, 3.18097166e+02], [0.00000000e+00, 2.23333187e+03, 2.35140624e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        self.d1 = [[ 8.30552296e-02,  6.51833604e+01,  1.11158762e-02,  9.54176741e-02, -3.86416392e+03]]
        self.d2 = [[-1.11236940e+00,  2.47020346e+02, -7.31266691e-03,  1.10678038e-01, 9.51622560e-01]]
        self.R = [[ 0.99974767,  0.01893509,  0.01208513], [-0.01912375,  0.99969399,  0.01569099], [-0.01178432, -0.01591815,  0.99980385]]

        self.T = [[-2.89122739], [-1.56384546], [33.90900759]]

stereo = shuangmu()

class stereoCameral(object):
    def __init__(self):

        self.cam_matrix_left = stereo.m1

        self.cam_matrix_right = stereo.m2

        #[k1, k2, p1, p2, k3]
        self.distortion_l = stereo.d1
        self.distortion_r = stereo.d2


        self.R = stereo.R

        self.T = stereo.T

        self.baseline = stereo.T[0]



def preprocess(img1, img2):

    im1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    im1 = cv2.equalizeHist(im1)
    im2 = cv2.equalizeHist(im2)

    return im1, im2



def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image




def getRectifyTransform(height, width, config):

    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T


    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0.8)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_16SC2)
    print(width, height)

    return map1x, map1y, map2x, map2y, Q



def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    return rectifyed_img1, rectifyed_img2



def draw_line1(image1, image2):

    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output



def draw_line2(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    for k in range(15):
        cv2.line(output, (0, 50 * (k + 1)), (2 * width, 50 * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output



def disparity_SGBM(left_image, right_image, down_scale=False):

    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    param = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }


    sgbm = cv2.StereoSGBM_create(**param)


    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = sgbm.compute(left_image, right_image)
        disparity_right = sgbm.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = size[0] / left_image_down.shape[1]
        disparity_left_half = sgbm.compute(left_image_down, right_image_down)
        disparity_right_half = sgbm.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left *= factor
        disparity_right *= factor

    return disparity_left, disparity_right

if __name__ == '__main__':
    #     calibration_photo()
    biaoding = StereoCalibration()
    biaoding.calibration_photo()
    imgL = cv2.imread("imgL/imgL_1.jpg")
    imgR = cv2.imread("imgR/imgR_1.jpg")


    height, width = 480,640
    config = stereoCameral()


    #imgL = undistortion(imgL, config.cam_matrix_left, config.distortion_l)
    #imgR = undistortion(imgR, config.cam_matrix_right, config.distortion_r)


    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    iml_rectified, imr_rectified = rectifyImage(imgL, imgR, map1x, map1y, map2x, map2y)
    plt.imshow(imgL)
    plt.show()
    plt.imshow(imgR)
    plt.show()
    linepic = draw_line1(iml_rectified, imr_rectified)
    plt.imshow(linepic)
    plt.show()

    lookdispL, lookdispR = disparity_SGBM(iml_rectified, imr_rectified)
    linepic2 = draw_line2(lookdispL, lookdispR)

    plt.imshow(linepic2)
    plt.show()

