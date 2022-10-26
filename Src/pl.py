from pprint import pprint
from copy import deepcopy
from typing import List
from loguru import logger

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core_func import *

class config:
    debugShowImg = False
    debugLastonly = False

    GaussianBlurSize = 5

    MorphSizeWidth = 17
    MorphSizeHeight = 3

    ByColorAdaptive = True
    ByColorAdaptiveSV = 64
    ByColorMaxSV = 255
    ByColorMinSV = 95
    ByColorBlue = (100,140)
    ByColorYellow = (15,40)

    MaxDeskewAngle = 60

    PlateSize = (136, 36)

class Img:
    img = None
    org_img = None
    contoursData = []
    
    def _debugShowImg(self,img,title):
        if config.debugShowImg and not config.debugLastonly:
            cv2.imshow(title, img)
    
    def _drawContours(self,contours):
        r = deepcopy(self.org_img)
        cv2.drawContours(r, [i[0] for i in contours], -1, (128,0,255), 3)
        return r

    def init(self,img,org_img = None):
        self.img = img
        if org_img is not None:
            self.org_img = org_img
        return self
    
    def GaussianBlur(self, size):
        r = cv2.GaussianBlur(self.img, (size, size), 0)
        #self._debugShowImg(r,"GaussianBlur")
        return r

    def imgToGray(self):
        r = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        #self._debugShowImg(r,"imgToGray")
        return r

    def Laplace(self):
        return cv2.Laplacian(self.img, cv2.CV_64F)

    def Sobel(self):
        # x y 方向
        x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        y = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        
        # 格式转换
        Scale_absX = cv2.convertScaleAbs(x)  
        Scale_absY = cv2.convertScaleAbs(y)

        # 图像混合
        r = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0) 
        #self._debugShowImg(r,"Sobel")
        return r

    def Shcarr(self):
        return cv2.Scharr(self.img, cv2.CV_64F, 0, 1)

    def threshold(self):
        ret , binary = cv2.threshold(self.img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #self._debugShowImg(binary,"threshold")
        return binary
    
    def morphology(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.MorphSizeWidth, config.MorphSizeHeight))
        r = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        #self._debugShowImg(r,"morphology")
        return r

    def findContours(self):
        # 轮廓检测
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #self._debugShowImg(self._drawContours(contours),"findContours")
        return contours

    def getMinAreaRect(self,contours):

        self.contoursData = [
            (
                np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
                , cv2.minAreaRect(contour)
            )
            for contour in contours
        ]
        self._debugShowImg(self._drawContours(self.contoursData),"getMinAreaRect")
        return self.contoursData

    def findByColor(self,Tcolor="Blue"):
        minC , maxC = 0 , 0
        if Tcolor == "Blue":
            minC = config.ByColorBlue[0]
            maxC = config.ByColorBlue[1]
        elif Tcolor == "Yellow":
            minC = config.ByColorYellow[0]
            maxC = config.ByColorYellow[1]
        else:
            logger.error("Color not select")
            return None
        diffC = (maxC - minC)/2
        avgC = minC + diffC

        # 转换到HSV
        hsvimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hsvSplit = cv2.split(hsvimg)
        cv2.equalizeHist(hsvSplit[2],hsvSplit[2])
        hsvimg = cv2.merge(hsvSplit)

        s_all = 0
        v_all = 0
        count = 0
        for i in range(hsvimg.shape[0]):
            for j in range(hsvimg.shape[1]):
                H = hsvimg[i, j, 0]
                S = hsvimg[i, j, 1]
                V = hsvimg[i, j, 2]

                s_all += S
                v_all += V
                count += 1
                colorMatched = False

                if H > minC and H < maxC:
                    if H > avgC:
                        Hdiff = H - avgC
                    else:
                        Hdiff = avgC - H

                    Hdiff_p = Hdiff / diffC

                    min_sv = 0

                    if config.ByColorAdaptive:
                        min_sv = config.ByColorAdaptiveSV - config.ByColorAdaptiveSV / 2 * (1 - Hdiff_p)
                    else:
                        min_sv = config.ByColorMinSV

                    if (S > min_sv and S < config.ByColorMaxSV) and (V > min_sv and V < config.ByColorMaxSV):
                        colorMatched = True
                if colorMatched:
                    hsvimg[i, j, 0] = 0
                    hsvimg[i, j, 1] = 0
                    hsvimg[i, j, 2] = 255
                else:
                    hsvimg[i, j, 0] = 0
                    hsvimg[i, j, 1] = 0
                    hsvimg[i, j, 2] = 0

        r = cv2.split(hsvimg)[2]
        #self._debugShowImg(r,"findByColor")
        return r
        
    def verifyContours(self):
        def filterContoursBySize(contours):
            # 筛选面积小的
            r = []
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour[0])
                # 面积太大的都筛选掉
                if area < 1000:
                    continue
                r.append(contour)
            return r
        def filterContoursByAngle(contours):
            # 筛选角度
            r = []
            for contour in contours:
                rect = contour[1]
                angle = rect[2]
                scale = rect[1][0] / rect[1][1]
                if scale < 1:
                    angle -= 90
                print("angle:",angle,"scale:",scale)
                
                

                if -config.MaxDeskewAngle < angle < config.MaxDeskewAngle:
                    r.append(contour)
            return r
        self.contoursData = filterContoursByAngle(filterContoursBySize(self.contoursData))
        self._debugShowImg(self._drawContours(self.contoursData),"verifyContours")
        return self.contoursData 

    def deskews(self):
        plates = []
        counter = 0
        print(len(self.contoursData))
        #self.contoursData.pop(0)
        for contour , rect in self.contoursData:
            # 数据
            roi_angle = rect[2]
            roi_scale = rect[1][0] / rect[1][1]
            roi_size = rect[1]
            if roi_scale < 1:
                roi_angle -= 90
                roi_size = (rect[1][1], rect[1][0])
            
            # 获取ROI区域
            (x, y, w, h), flag = self.calcSafeRect(rect, self.org_img)
            if not flag:
                continue
            bound_mat = self.org_img[y: y + h, x: x + w, :]
            #cv2.imshow(f"bound_mat {counter}",bound_mat)
            bound_mat_b = self.img[y: y + h, x: x + w]
            roi_ref_center = (rect[0][0] - x, rect[0][1] - y)

            # 是否需要旋转
            if (-3 < roi_angle < 3) or roi_angle == 90 or roi_angle == -90:
                deskew_mat = bound_mat
            else:
                # 旋转
                rotated_mat, flag = self.rotation(bound_mat, roi_size, roi_ref_center,roi_angle)
                if not flag:
                    continue
                #cv2.imshow(f"rotated_mat {counter}",rotated_mat)

                rotated_mat_b, flag = self.rotation2(bound_mat_b, roi_size, roi_ref_center,roi_angle)
                if not flag:
                    continue
                
                # 仿射变换
                roi_slope, flag = self.isdeflection(rotated_mat_b, roi_angle)
                if flag:
                    #print("need slope")
                    deskew_mat = self.affine(rotated_mat, roi_slope)
                else:
                    deskew_mat = rotated_mat
                #cv2.imshow(f"deskew_mat {counter}",deskew_mat)
            
            # 保存
            if deskew_mat.shape[0] >= 36 or deskew_mat.shape[1] >= 136:
                plate_mat = cv2.resize(deskew_mat, config.PlateSize, interpolation=cv2.INTER_AREA)
            else:
                plate_mat = cv2.resize(deskew_mat, config.PlateSize, interpolation=cv2.INTER_CUBIC)

            p:Plate = Plate()
            p.plateImage = plate_mat
            p.platePos = rect
            plates.append(p)

            counter += 1
        return plates

    def verifySizes(self, mr):
        if mr[1][0] == 0 or mr[1][1] == 0:
            return False

        # life mode
        error = self.m_error
        aspect = self.m_aspect
        min = 34 * 8 * self.m_verifyMin
        max = 34 * 8 * self.m_verifyMax

        rmin = aspect - aspect * error
        rmax = aspect + aspect * error

        area = mr[1][0] * mr[1][1]  # height * width
        r = mr[1][0] / mr[1][1]

        if r < 1:
            r = 1 / r

        if (area < min or area > max) or (r < rmin or r > rmax):
            return False
        else:
            return True

    def calcSafeRect(self, roi, src):
        '''
            return [x, y, w, h]
        '''
        box = cv2.boxPoints(roi)
        x, y, w, h = cv2.boundingRect(box)

        src_h, src_w, _ = src.shape

        tl_x = x if x > 0 else 0
        tl_y = y if y > 0 else 0
        br_x = x + w - 1 if x + w - 1 < src_w else src_w - 1
        br_y = y + h - 1 if y + h - 1 < src_h else src_h - 1

        roi_w = br_x - tl_x
        roi_h = br_y - tl_y
        if roi_w <= 0 or roi_h <= 0:
            return [tl_x, tl_y, roi_w, roi_h], False

        return [tl_x, tl_y, roi_w, roi_h], True

    def rotation(self, in_img, rect_size, center, angle):
        '''
            rect_size: (h, w)
            rotation an image
        '''
        if len(in_img.shape) == 3:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5), 3)).astype(in_img.dtype)
        else:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5))).astype(in_img.dtype)

        x = int(max(in_large.shape[1] / 2 - center[0], 0))
        y = int(max(in_large.shape[0] / 2 - center[1], 0))

        width = int(min(in_img.shape[1], in_large.shape[1] - x))
        height = int(min(in_img.shape[0], in_large.shape[0] - y))

        if width != in_img.shape[1] and height != in_img.shape[0]:
            return in_img, False

        t_roi = in_large[y: y + height, x: x + width]
        cv2.addWeighted(t_roi,0,in_img,1,0,t_roi)
        #cv2.imshow("p123",in_large)

        new_center = (in_large.shape[1] / 2, in_large.shape[0] / 2)

        rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1)

        mat_rotated = cv2.warpAffine(in_large, rot_mat, (in_large.shape[1], in_large.shape[0]), cv2.INTER_CUBIC)
        #cv2.imshow("p124",mat_rotated)
        img_crop = cv2.getRectSubPix(mat_rotated, (int(rect_size[0]), int(rect_size[1])), new_center)
        return img_crop, True

    def rotation2(self, in_img, rect_size, center, angle):
        '''
            rect_size: (h, w)
            rotation an image
        '''
        if len(in_img.shape) == 3:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5), 3)).astype(in_img.dtype)
        else:
            in_large = np.zeros((int(in_img.shape[0] * 1.5), int(in_img.shape[1] * 1.5))).astype(in_img.dtype)

        x = int(max(in_large.shape[1] / 2 - center[0], 0))
        y = int(max(in_large.shape[0] / 2 - center[1], 0))

        width = int(min(in_img.shape[1], in_large.shape[1] - x))
        height = int(min(in_img.shape[0], in_large.shape[0] - y))

        if width != in_img.shape[1] and height != in_img.shape[0]:
            return in_img, False

        t_roi = in_large[y: y + height, x: x + width]
        cv2.addWeighted(t_roi,0,in_img,1,0,t_roi)
        #cv2.imshow("p123",in_large)

        new_center = (in_large.shape[1] / 2, in_large.shape[0] / 2)

        rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1)

        mat_rotated = cv2.warpAffine(in_large, rot_mat, (in_large.shape[1], in_large.shape[0]), cv2.INTER_CUBIC)
        #cv2.imshow("p124",mat_rotated)
        img_crop = cv2.getRectSubPix(mat_rotated, (int(rect_size[0]), int(rect_size[1])), new_center)
        return img_crop, True

    def affine(self, in_mat, slope):
        height = in_mat.shape[0]
        width = in_mat.shape[1]
        xiff = abs(slope) * height
        if slope > 0:
            plTri = np.float32([[0, 0], [width - xiff - 1, 0], [xiff, height - 1]])
            dstTri = np.float32([[xiff / 2, 0], [width - 1 - xiff / 2, 0], [xiff / 2, height - 1]])
        else:
            plTri = np.float32([[xiff, 0], [width - 1, 0], [0, height - 1]])
            dstTri = np.float32([[xiff / 2, 0], [width - 1 - xiff / 2, 0], [xiff / 2, height - 1]])
        warp_mat = cv2.getAffineTransform(plTri, dstTri)

        if in_mat.shape[0] > 36 or in_mat.shape[1] > 136:
            affine_mat = cv2.warpAffine(in_mat, warp_mat, (int(width),int(height)), cv2.INTER_AREA)
        else:
            affine_mat = cv2.warpAffine(in_mat, warp_mat, (int(width),int(height)), cv2.INTER_CUBIC)
        return affine_mat

    def isdeflection(self, in_img, angle):
        comp_index = [in_img.shape[0] / 4, in_img.shape[0] / 2, in_img.shape[0] / 4 * 3]
        len = []
        for i in range(3):
            index = comp_index[i]
            j = 0
            value = 0
            while value == 0 and j < in_img.shape[1]:
                value = in_img[int(index), j]
                j += 1
            len.append(j)
        maxlen = max(len[2], len[0])
        minlen = min(len[2], len[0])
        difflen = abs(len[2] - len[0])
        PI = 3.1415926
        import math
        g = math.tan(angle * PI / 180)

        if (maxlen - len[1] > (in_img.shape[1] / 32)) or (len[1] - minlen > (in_img.shape[1] / 32)):
            slope_can_1 = (len[2] - len[0]) / comp_index[1]
            slope_can_2 = (len[1] - len[0]) / comp_index[0]
            slope_can_3 = (len[2] - len[1]) / comp_index[0]
            slope = slope_can_1 if abs(slope_can_1 - g) <= abs(slope_can_2 - g) else slope_can_2
            return slope, True
        else:
            slope = 0
        return slope, False

class Plate:
    plateTypeColor = ""
    plateImage = None
    plateString = ""
    platePos = None
    
    def stringSplit(self):
        chars = []
        CharsSegment().charsSegment(self.plateImage,chars)
        # for i,e in enumerate(chars):
        #     plt.subplot(5,5,i+1)
        #     plt.grid(False)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(e,cmap=plt.cm.binary)
        # plt.show()
        

class PlateLocate:

    def mian(self,imgData):
        i = Img().init(imgData)
        # blur
        i.img = i.GaussianBlur(5)
        # sobel
        SobelImg = Img().init(i.imgToGray(),i.img)
        SobelImg.img = SobelImg.Sobel()
        # Contours
        # SobelImg.img = SobelImg.threshold()
        # SobelImg.img = SobelImg.morphology()
        # contoursData = SobelImg.findContours()
        # contoursData = SobelImg.getMinAreaRect(contoursData)
        # contoursData = SobelImg.verifyContours(contoursData)
        # Color
        ColorImg = Img().init(i.img,i.img)
        ColorImg.img = ColorImg.findByColor("Blue")
        ColorImg.img = ColorImg.threshold()
        ColorImg.img = ColorImg.morphology()
        contoursData = ColorImg.findContours()
        ColorImg.getMinAreaRect(contoursData)
        ColorImg.verifyContours()
        #ColorImg.getArea(contoursData)

        plates =  ColorImg.deskews()
        return plates


class CharsSegment(object):
    def __init__(self):
        self.LiuDingSize = 7
        self.MatWidth = 136

        self.colorThreshold = 150
        self.BluePercent = 0.3
        self.WhitePercent = 0.1

        self.m_debug = True

    def verifyCharSizes(self, r):
        aspect = 0.5
        charAspect = r.shape[1] / r.shape[0]
        error = 0.7
        minH = 10
        maxH = 35

        minAspect = 0.05  # for number 1
        maxAspect = aspect + aspect * error

        area = cv2.countNonZero(r)
        bbArea = r.shape[0] * r.shape[1]
        percPixels = area / bbArea

        if percPixels <= 1 and minAspect < charAspect < maxAspect and minH <= r.shape[0] < maxH:
            return True
        else:
            return False

    def preprocessChar(self, in_mat):
        h = in_mat.shape[0]
        w = in_mat.shape[1]

        charSize = 20
        transform = np.array([[1, 0, 0],
                              [0, 1, 0]], dtype=np.float32)
        m = max(w, h)
        transform[0][2] = m / 2 - w / 2
        transform[1][2] = m / 2 - h / 2

        warpImage = cv2.warpAffine(in_mat, transform, (m, m), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

        return cv2.resize(warpImage, (charSize, charSize))

    def charsSegment(self, input, result):
        w = input.shape[1]
        h = input.shape[0]

        tmp = input[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

        plateType = getPlateType(tmp, True)

        input_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        if plateType == Color.BLUE:

            w = input_gray.shape[1]
            h = input_gray.shape[0]

            tmp = input_gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

            threadHoldV = ThresholdOtsu(tmp)

            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY)
        elif plateType == Color.YELLOW:
            w = input_gray.shape[1]
            h = input_gray.shape[0]

            tmp = input_gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

            threadHoldV = ThresholdOtsu(tmp)

            _, img_threshold = cv2.threshold(input_gray, threadHoldV, 255, cv2.THRESH_BINARY_INV)
        elif plateType == Color.WHITE:
            _, img_threshold = cv2.threshold(input_gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        else:
            _, img_threshold = cv2.threshold(input_gray, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        if not clearLiuDingChar(img_threshold):
            return 2

        img_contours = img_threshold.copy()
        #cv2.imshow("img_contours",img_contours)

        contours, _ = cv2.findContours(img_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        r = deepcopy(input)
        cv2.drawContours(r, [np.int0(cv2.boxPoints(cv2.minAreaRect(contour))) for contour in contours], -1, (128,0,255), 3)
        #cv2.imshow("sContours", r)

        vecRect = []
        for it in contours:

            mr = cv2.boundingRect(it)
            x, y, w, h = map(int, mr)

            roi = img_threshold[y:y + h, x:x + w]
            if self.verifyCharSizes(roi):
                vecRect.append(mr)

        if len(vecRect) == 0:
            return 3

        vecRect = sorted(vecRect, key=lambda x: x[0])

        specIndex = self.GetSpecificRect(vecRect)

        if specIndex < len(vecRect):
            chineseRect = self.GetChineseRect(vecRect[specIndex])
        else:
            return 4

        newSorted = []
        newSorted.append(chineseRect)
        self.RebuildRect(vecRect, newSorted, specIndex)
        if len(newSorted) == 0:
            return 5

        for mr in newSorted:
            x, y, w, h = map(int, mr)
            auxRoi = input_gray[y:y + h, x:x + w]
            if plateType == Color.BLUE:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif plateType == Color.YELLOW:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif plateType == Color.WHITE:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
            else:
                _, newroi = cv2.threshold(auxRoi, 5, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

            newroi = self.preprocessChar(newroi)

            result.append(newroi)
        return 0

    def GetSpecificRect(self, vecRect):
        xpos = []
        maxH = 0
        maxW = 0
        for rect in vecRect:  # (x, y, w, h)
            xpos.append(rect[0])

            if rect[3] > maxH:
                maxH = rect[3]
            if rect[2] > maxW:
                maxW = rect[2]
        specIndex = 0
        for i in range(len(vecRect)):
            mr = vecRect[i]
            midx = mr[0] + mr[2] / 2
            if (mr[2] > maxW * 0.8 or mr[3] > maxH * 0.8) and \
                                    int(self.MatWidth / 7) < midx < 2 * int(self.MatWidth / 7):
                specIndex = i
        return specIndex

    def GetChineseRect(self, rectSpe):
        h = rectSpe[3]
        newW = rectSpe[2] * 1.15
        x = rectSpe[0]
        y = rectSpe[1]

        newX = x - int(newW * 1.15)
        newX = newX if newX > 0 else 0

        return (newX, y, int(newW), h)

    def RebuildRect(self, vecRect, outRect, specIndex):
        count = 6
        for i in range(specIndex, len(vecRect)):
            if count == 0:
                break
            outRect.append(vecRect[i])
            count -= 1

def workflow(name):
    # read img
    img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    plates:List[Plate] = PlateLocate().mian(img)
    print(plates)
    
    rl = []
    #res = PlateJudge().judge(plates, r"models/plate_detect")
    for i,plate in enumerate(plates):
        #print("plate",plate.plateImage)
        #cv2.imshow(f"plate{i}", plate.plateImage)
        #plate.stringSplit()
        rl.append(plate.plateImage)
    return rl

if __name__ == "__main__":
    workflow("Imgs/plate_locate.jpg")