#!/usr/bin/env python3

import sys
import os
from os import path
import numpy as np
import imageio
import cv2
import glob
import skimage
from skimage import filters
from skimage import morphology
from skimage import measure


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_val = skimage.filters.threshold_sauvola(gray, window_size=159)
    bin_img = (gray > threshold_val).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    blur = cv2.bilateralFilter(bin_img, 10, 75, 75)
    inv = np.invert(blur)
    img_dilate = cv2.dilate(inv, kernel, iterations=6)
    img_erode = cv2.erode(img_dilate, kernel, iterations=3)
    inv_img = np.invert(img_erode)
    all_labels = skimage.measure.label(1 - skimage.morphology.remove_small_objects((1 - inv_img).astype(bool),min_size=5000))
    kernel = np.ones((5,5), np.uint8)
    img_dilate = cv2.dilate(all_labels.astype(np.uint8), kernel, iterations=6) 
    return img_dilate

def intersect(img):
    height, width = img.shape
    step = 60
    intersects = np.full(img.shape, 0)
    for i in range(0, height-step, step//2):
        for j in range(0, width-step, step//2):
            colors = set()
            for k in range(i, i+step):
                for l in range(j, j+step):
                    pixel = img[k][l]
                    if pixel != -1 and pixel not in colors:
                        colors.add(pixel)
            if len(colors) >= 4:
                intersects[i+step//2][j+step//2] = 1


    # mean
    step = 80
    intersects2 = np.full(img.shape, 0)
    for i in range(0, height-step, step):
        for j in range(0, width-step, step):
            pts = []
            for k in range(i, i+step):
                for l in range(j, j+step):
                    if intersects[k][l] != 0:
                        pts.append((k, l))
            if len(pts) >= 2:
                sum_x = 0
                sum_y = 0
                for point in pts:
                    sum_y += point[0]
                    sum_x += point[1]
                mean = (sum_y // len(pts), sum_x // len(pts))
                intersects2[mean[0]][mean[1]] = 1
            else:
                for point in pts:
                    intersects2[point[0]][point[1]] = 1


    # mean
    step = 100
    intersects3 = np.full(img.shape, 0)
    for i in range(0, height-step, step):
        for j in range(0, width-step, step):
            pts = []
            for k in range(i, i+step):
                for l in range(j, j+step):
                    if intersects2[k][l] != 0:
                        pts.append((k, l))
            if len(pts) >= 2:
                sum_x = 0
                sum_y = 0
                for point in pts:
                    sum_y += point[0]
                    sum_x += point[1]
                mean = (sum_y // len(pts), sum_x // len(pts))
                intersects3[mean[0]][mean[1]] = 1
            else:
                for point in pts:
                    intersects3[point[0]][point[1]] = 1


    step = 85
    res = [] # list of tuple
    for i in range(0, height-step, step):
        for j in range(0, width-step, step):
            pts = []
            for k in range(i, i+step):
                for l in range(j, j+step):
                    if intersects3[k][l] != 0:
                        pts.append((k, l))
            if len(pts) >= 2:
                sum_x = 0
                sum_y = 0
                for point in pts:
                    sum_y += point[0]
                    sum_x += point[1]
                res.append((sum_y // len(pts), sum_x // len(pts)))
            else:
                for point in pts:
                    res.append(point)
    return res


def main():
    # Load all images from TRAIN
    if not os.path.exists('results'):
        os.makedirs('results')
    for filename in glob.glob(sys.argv[1] + '/*.jpg'):
        img = imageio.imread(filename)
        labels = process(img)
        points = intersect(labels)
        np.savetxt('results/' + filename[len(sys.argv[1]):-4] + '.csv', points, delimiter=',')


if __name__ == "__main__":
    main()
