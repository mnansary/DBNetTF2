#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from cv2 import resize
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as KL
from termcolor import colored
import cv2 
import math
import numpy as np 
import matplotlib.pyplot as plt 
import pyclipper
from shapely.geometry import Polygon

#--------------------------helper---------------------------------

#----------------------------------detector------------------------
class Detector(object):
    def __init__(self,
                weight_path,
                k=50,
                dim=512,
                outs=["conv2_block3_out",
                "conv3_block4_out",
                "conv4_block6_out",
                "conv5_block3_out"]):
        '''
            initializes a dbnet detector model
            args:
                weight_path :   path to .h5 weight
        '''
        self.mean = np.array([103.939, 116.779, 123.68])
        self.dim=dim
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        with strategy.scope():
            self.model=self.net(k=k,dim=dim,outs=outs)
            self.model.load_weights(weight_path)
            print(colored("#LOG     :","green"),colored("weights loaded!","blue"))
    
    def resize(self,size, image,pad):
        h, w = image.shape[0],image.shape[1]
        scale_w = size / w
        scale_h = size / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        if len(image.shape)==3:
            padimg = np.ones((size, size, 3), image.dtype)*pad
        else:
            padimg = np.ones((size, size), image.dtype)*pad
        padimg[:h, :w] = cv2.resize(image, (w, h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        return np.squeeze(padimg)
            
    def box_score_fast(self,bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


    def unclip(self,box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self,contour):
        if not contour.size:
            return [], 0
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
            points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def polygons_from_bitmap(self,pred, bitmap, dest_width, dest_height, max_candidates=500, box_thresh=0.5):
        pred = pred[..., 0]
        bitmap = bitmap[..., 0]
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:max_candidates]:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            
            if points.shape[0] < 4:
                continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue
            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=2.0)
                if len(box) > 1:
                    continue
            else:
                continue

            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < 5:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores



            
    def net(self,
            k=50,
            dim=512,
            outs=["conv2_block3_out",
                    "conv3_block4_out",
                    "conv4_block6_out",
                    "conv5_block3_out"]):
        # input layer
        input_image = KL.Input(shape=[dim,dim, 3], name='image')
        backbone = K.applications.resnet50.ResNet50(input_tensor=input_image,weights=None,include_top=False)
        C2, C3, C4, C5 = [backbone.get_layer(out).output for out in outs]

        # in2
        in2 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
        in2 = KL.BatchNormalization()(in2)
        in2 = KL.ReLU()(in2)
        # in3
        in3 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
        in3 = KL.BatchNormalization()(in3)
        in3 = KL.ReLU()(in3)
        # in4
        in4 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
        in4 = KL.BatchNormalization()(in4)
        in4 = KL.ReLU()(in4)
        # in5
        in5 = KL.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)
        in5 = KL.BatchNormalization()(in5)
        in5 = KL.ReLU()(in5)

        # P5
        P5 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5)
        P5 = KL.BatchNormalization()(P5)
        P5 = KL.ReLU()(P5)
        P5 = KL.UpSampling2D(size=(8, 8))(P5)
        # P4
        out4 = KL.Add()([in4, KL.UpSampling2D(size=(2, 2))(in5)])
        P4 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4)
        P4 = KL.BatchNormalization()(P4)
        P4 = KL.ReLU()(P4)
        P4 = KL.UpSampling2D(size=(4, 4))(P4)
        # P3
        out3 = KL.Add()([in3, KL.UpSampling2D(size=(2, 2))(out4)])
        P3 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3)
        P3 = KL.BatchNormalization()(P3)
        P3 = KL.ReLU()(P3)
        P3 = KL.UpSampling2D(size=(2, 2))(P3)
        # P2
        out2 = KL.Add()([in2, KL.UpSampling2D(size=(2, 2))(out3)])
        P2 = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out2)
        P2 = KL.BatchNormalization()(P2)
        P2 = KL.ReLU()(P2)

        fuse = KL.Concatenate()([P2, P3, P4, P5])

        # binarize map
        p = KL.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        p = KL.BatchNormalization()(p)
        p = KL.ReLU()(p)
        p = KL.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
        p = KL.BatchNormalization()(p)
        p = KL.ReLU()(p)
        binarize_map  = KL.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',activation='sigmoid', name='binarize_map')(p)
        db_model = K.Model(inputs=input_image,outputs=binarize_map)
        return db_model
    
    def detect(self,img,debug=False):
        if type(img)==str:
            img=cv2.imread(img)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h_org,w_org,_=img.shape
        if debug:
            plt.imshow(img)
            plt.show()
        
        data=self.resize(self.dim,img,0)
        data = data.astype(np.float32)
        data-= self.mean
        data = data/255
        if debug:
            plt.imshow(data)
            plt.show()
        data=np.expand_dims(data,axis=0)
        pred=self.model.predict(data)[0]
        if debug:
            plt.imshow(pred)
            plt.show()
        bitmap = pred > 0.3
        boxes, _ = self.polygons_from_bitmap(pred, bitmap, w_org, h_org)
        #------------------------------
        # TODO: Poly boxes to crops
        #------------------------------
        