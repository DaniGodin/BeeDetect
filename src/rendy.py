#!/usr/bin/env python3

import sys
import os
from os import path
import numpy as np
import imageio
import cv2
import glob


def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    retval, markers = cv2.connectedComponents(opening)
    labels = cv2.watershed(img,markers)
    return labels


def intersect(img):
    height, width = img.shape
    step = 5
    points = []
    for i in range(0, height-step, step):
        for j in range(0, width-step, step):
            colors = set()
            for k in range(i, i+step):
                for l in range(j, j+step):
                    pixel = img[k][l]
                    if pixel != -1 and pixel not in colors:
                        colors.add(pixel)
            if len(colors) >= 3:
                points.append((i+step//2, j+step//2))
    return points



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
