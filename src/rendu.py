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


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = gray
    for i in range(20):
        blur = cv2.GaussianBlur(blur,(5,5),0)
    threshold_val = skimage.filters.threshold_sauvola(blur, window_size=159)
    bin_img = (blur > threshold_val).astype(np.uint8)
    all_labels = skimage.measure.label(1 - skimage.morphology.remove_small_objects((1 - bin_img).astype(bool),min_size=5000))
    kernel = np.ones((5,5), np.uint8)
    img_dilate = cv2.dilate(all_labels.astype(np.uint8), kernel, iterations=3) 
    return img_dilate

def intersect(img):
    height, width = img.shape
    step = 60
    intersects = np.full(img.shape, 0)
    for i in range(0, height-step, step):
        for j in range(0, width-step, step):
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
    res = [] # list of tuple
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
