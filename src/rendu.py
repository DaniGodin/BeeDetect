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
    threshold_val = skimage.filters.threshold_sauvola(gray, window_size=129, k=0.05)
    bin_img = (gray > threshold_val).astype(np.uint8)
    inv = np.invert(bin_img)
    kernel = np.ones((3,3), np.uint8)
    closing = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    inv_img = np.invert(closing)
    removed = skimage.morphology.remove_small_objects((1 - inv_img).astype(bool),min_size=15000)
    removed2 = skimage.morphology.remove_small_objects((1 - removed).astype(bool), min_size=5000)
    ret, all_labels = cv2.connectedComponents(removed2.astype(np.uint8))
    return all_labels

def intersect(img):
    height, width = img.shape
    step = 60
    intersects = np.full(img.shape, 0)
    for i in range(0, height-step, step//2):
        for j in range(0, width-step, step//2):
            colors = {} # number of pixel per colors
            for k in range(i, i+step):
                for l in range(j, j+step):
                    pixel = img[k][l]
                    if pixel != 0:
                        if pixel not in colors:
                            colors[pixel] = 1
                        else:
                            colors[pixel] += 1
            nb_colors = 0
            for color in colors:
                if colors[color] > 15:
                    nb_colors += 1
            if nb_colors >= 3:
                intersects[i+step//2][j+step//2] = 1

    # mean
    step = 100
    res = set() # set of tuple
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
                res.add(mean)
            else:
                for point in pts:
                    intersects2[point[0]][point[1]] = 255
                    res.add(point)

    # mean
    step = 100
    res = set() # set of tuple
    for i in range(50, height-step, step):
        for j in range(50, width-step, step):
            # print grid
            for i2 in range(0, 5):
                for j2 in range(0, 5):
                    img[i + i2][j + j2] = 255
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
                res.add(mean)
            else:
                for point in pts:
                    res.add(point)
    return res


def main():
    # Load all images from TRAIN
    if not os.path.exists('results'):
        os.makedirs('results')
    for filename in glob.glob(sys.argv[1] + '/*.jpg'):
        img = imageio.imread(filename)
        labels = process(img)
        points = intersect(labels)
        np.savetxt('results/' + filename[len(sys.argv[1]):-4] + '.csv', np.array(list(points)), delimiter=',')


if __name__ == "__main__":
    main()
