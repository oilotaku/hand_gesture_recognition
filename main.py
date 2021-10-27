# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from camera_L import cameradataL
from camera_R import cameradataR
from SGB import disparity_SGBM
from tensorflow.keras import models

cameraR = cv2.VideoCapture(1)
cameraL = cv2.VideoCapture(2)

pathR = './imgR/'
pathL = './imgL/'

fnameR = 'imgR_'
fnameL = 'imgL_'

stime = 0
model = models.load_model('cnn_model.h5')
#model_RGB = models.load_model('cnn_model_RGB.h5')
def imgin():
    i = 100

    while True:
        ret, imgR = cameraR.read()
        ret, imgL = cameraL.read()

        cv2.imshow('imgR', imgR)
        cv2.imshow('imgL', imgL)



        if cv2.waitKey(1) & 0xFF == ord('s'):
            i += 1
            cv2.imwrite(pathR + fnameR + str(i) + '.jpg', imgR)
            cv2.imwrite(pathL + fnameL + str(i) + '.jpg', imgL)
            print('save:' + pathR + fnameR + str(i) + '.jpg' + pathL + fnameL + str(i) + '.jpg')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraL.release()
    cameraR.release()
    cv2.destroyWindow("imgR")
    cv2.destroyWindow("imgL")

def stereoCalibrate_camera():
    frameR = cv2.imread('./imgR/imgR_1.jpg')
    frameRG = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00000001)

    npcmpR = np.load('cmpR.npz')
    npcmpL = np.load('cmpL.npz')

    objpointsL = npcmpL['objpointsL']
    objpointsR = npcmpR['objpointsR']
    imgpointsR = npcmpR['imgpointsR']
    imgpointsL = npcmpL['imgpointsL']
    mtxL = npcmpL['mtxL']
    mtxR = npcmpR['mtxR']
    distL = npcmpL['distL']
    distR = npcmpR['distR']

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    flags |= cv2.CALIB_ZERO_TANGENT_DIST

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpointsR, imgpointsR, imgpointsL, mtxR, distR, mtxL,
                                                               distL, frameRG.shape[::-1], criteria_stereo, flags)
    # print(retS, MLS, dLS, MRS, dRS, R, T, E, F)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, frameRG.shape[::-1], R, T, flags=flags,
                                                      alpha=0, newImageSize=(0, 0))

    # print('RL, RR, PL, PR, Q, roiL, roiR', RL, RR, PL, PR, Q, roiL, roiR)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                  frameRG.shape[::-1], cv2.CV_16SC2)

    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                   frameRG.shape[::-1], cv2.CV_16SC2)
    return Left_Stereo_Map, Right_Stereo_Map

def rectified(capL, cpaR):
    # frameR = cv2.imread('./imgR/imgR_101.jpg', 0)
    # frameL = cv2.imread('./imgL/imgL_101.jpg', 0)
    # frameR = cpaR.read()
    # frameL = capL.read()

    #Left_rectified = cv2.remap(capL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LINEAR)
    #im_L = Image.fromarray(Left_rectified)

    #Right_rectified = cv2.remap(cpaR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LINEAR)
    #im_R = Image.fromarray(Right_rectified)
    Left_rectified, Right_rectified = disparity_SGBM(capL, cpaR)

    return Left_rectified, Right_rectified

def skinmask(roi):
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 轉換至YCrCb空間
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (7, 7), 0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu處理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    ker = np.ones((5, 5), np.uint8)
    res = cv2.dilate(res, ker, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # 腐蝕影像
    res = cv2.erode(res, kernel)

    return res

def findCon(img, rimg):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    con, hie = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(rimg, con, -1, (255, 255, 255), 3)
    imgarr = np.array(gray, dtype=bool)
    rimg *= imgarr
    exist = (rimg != 0)
    mean = img.sum() / exist.sum()
    rimg = imgarr * mean
    return rimg

def point(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    con_frame = np.copy(th)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        hull = cv2.convexHull(contours[i])
        cv2.polylines(image, [hull], True, (0, 255, 0), 2)
    return image

Left_Stereo_Map, Right_Stereo_Map = stereoCalibrate_camera()

def cnn(img, img_d, switch=True):
    img = cv2.resize(img, (64, 48), interpolation=cv2.INTER_AREA)
    img_d = cv2.resize(img_d, (64, 48), interpolation=cv2.INTER_AREA)

    if switch:
        img = img.T
        img_d = img_d.T

        data = np.empty((48, 64, 4))
        data = data.T
        data[0] = img[0]
        data[1] = img[1]
        data[2] = img[2]
        data[3] = img_d
        data = data.T
        i_data = np.empty((1, 48, 64, 4))
        i_data[0] = data/255.0
        predictions = (model.predict(i_data) > 0.5)

    else:
        i_data = np.empty((1, 48, 64, 3))
        i_data[0] = img/255.0
        #predictions = (model_RGB.predict(i_data) > 0.5)

    return predictions

if __name__ == '__main__':
    i = 0
    while True:

        ret, imgR = cameraR.read()
        ret, imgL = cameraL.read()

        imgR = cv2.medianBlur(imgR, 11)
        imgL = cv2.medianBlur(imgL, 11)

        resL = skinmask(imgL)
        # resR = skinmask(imgR)

        Left_rectified, Right_rectified = rectified(resL, imgR)

        Left_rectified = findCon(resL, Left_rectified)

        predictions = cnn(resL, Left_rectified, switch=True)

        if predictions[0][0] == 0:
            if predictions[0][1] == 0:
                if predictions[0][2] == 0:
                    cv2.putText(imgL, 'no find', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)

        if predictions[0][0] == 1:
            cv2.putText(imgL, '1', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)

        if predictions[0][1] == 1:
            cv2.putText(imgL, '2', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)

        if predictions[0][2] == 1:
            cv2.putText(imgL, '3', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)

        ctime = time.time()
        fps = 1 / (ctime - stime)
        stime = ctime
        fps = int(fps)
        cv2.putText(imgL, 'FPS:' + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 4)
        # cv2.imshow('imgR', imgR)
        cv2.imshow('imgL', imgL)
        cv2.imshow('res', resL)

        # Left_rectified = cv2.Canny(Left_rectified, 100, 200)

        cv2.imshow('rectified', Left_rectified)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            i += 1
            cv2.imwrite('./train_D/' + str(i) + '.jpg', Left_rectified)
            cv2.imwrite('./train/' + str(i) + '.jpg', resL)
            print('./train/' + str(i) + '.jpg' + './train_D/' + str(i) + '.jpg')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraL.release()
    cameraR.release()
    # cv2.destroyWindow('imgR')
    cv2.destroyWindow('imgL')
    cv2.destroyWindow('res')
    cv2.destroyWindow('rectified')






