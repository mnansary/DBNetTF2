# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import cv2
import math
import numpy as np
import os.path as osp
import pyclipper
from shapely.geometry import Polygon


min_text_size=8
shrink_ratio=0.4

#--------------------
# maps
#--------------------
def draw_thresh_map(polygon, canvas, mask, shrink_ratio=0.4):
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded_polygon = np.array(padding.Execute(distance)[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = np.min(distance_map, axis=0)

    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        1 - distance_map[
            ymin_valid - ymin:ymax_valid - ymin+1,
            xmin_valid - xmin:xmax_valid - xmin+1],
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
    
def compute_distance(xs, ys, point_1, point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result

#--------------------
# data
#--------------------
def create_train_data(page):
    # unique 
    vals=[val for val in np.unique(page) if val>0]
    h,w=page.shape
    dim=(h,w)
    gt = np.zeros(dim, dtype=np.float32)
    mask = np.ones(dim, dtype=np.float32)
    thresh_map = np.zeros(dim, dtype=np.float32)
    thresh_mask = np.zeros(dim, dtype=np.float32)


    for v in vals:
        word_map=np.zeros_like(page)
        word_map[page==v]=255
        word_map=word_map.astype("uint8")
        # polygon
        contours, hiearchy = cv2.findContours(word_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cont = np.vstack(contours[i] for i in range(len(contours)))
        hull=cv2.convexHull(cont)
        poly=np.squeeze(hull)
        height = max(poly[:, 1]) - min(poly[:, 1])
        width = max(poly[:, 0]) - min(poly[:, 0])
        polygon = Polygon(poly)
        
        # generate gt and mask
        if polygon.area < 1 or min(height, width) < min_text_size:
            cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            continue
        else:
            distance = polygon.area * (1 - np.power(shrink_ratio, 2)) / polygon.length
            subject = [tuple(l) for l in poly]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if len(shrinked) == 0:
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                    cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                else:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
        # generate thresh map and thresh mask
        draw_thresh_map(poly, thresh_map, thresh_mask, shrink_ratio=shrink_ratio)
        
    gt*=255
    mask*=255
    thresh_map*=255
    thresh_mask*=255
    return gt,mask,thresh_map,thresh_mask