#!/usr/bin/env python

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
import pandas as pd
import numpy as np
import cv2 as cv
import random
import sys
import dlib
import imutils

def readPoints(path):
    '''Read points from .tem file'''
    # Create an array of points.
    points = []
    # Read points
    with open(path) as file:
        no_lines = int(file.readline())
        for i, line in enumerate(file):
            if 0 <= i < no_lines:
                x, y = line.split()
                points.append((int(float(x)), int(float(y))))

    return points


def applyAffineTransform(src, srcTri, dstTri, size):
    '''Apply affine transform calculated using srcTri and dstTri to src and output an image of size.'''
    # Given a pair of triangles, find the affine transform.
    warpMat = cv.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv.warpAffine(src, warpMat, (size[0], size[1]), None,
                        flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    return dst


def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    '''Warps and alpha blends triangular regions from img1 and img2 to img'''
    # Find bounding rectangle for each triangle
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))
    r = cv.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1] +
                                              r[3], r[0]:r[0]+r[2]] * (1 - mask) + imgRect * mask


def rect_contains(rect, point):
    '''Check if a point is inside a rectangle'''
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_point(img, p, color):
    '''Draw a point'''
    cv.circle(img, p, 2, color, cv.FILLED, cv.LINE_AA, 0)


def draw_voronoi(img, subdiv):
    '''Draw voronoi diagram'''
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))

        cv.fillConvexPoly(img, ifacet, color, cv.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv.polylines(img, ifacets, True, (0, 0, 0), 1, cv.LINE_AA, 0)
        cv.circle(img, (centers[i][0], centers[i][1]),
                  3, (0, 0, 0), cv.FILLED, cv.LINE_AA, 0)


def draw_delaunay(img, subdiv, delaunay_color):
    '''Draw delaunay triangles'''
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)


def calculateDelaunayTriangles(rect, subdiv, points, img, win_delaunay, delaunay_color, draw=False):
    '''Calculate delanauy triangle'''

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points (6 numbers)
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 0.1 and abs(pt[j][1] - points[k][1]) < 0.5):
                        ind.append(k)

            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

            # Draw lines
            if draw:
                cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
                cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
                cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)
                imgS = cv.resize(img, (413, 531))

    return delaunayTri

def drawLanmarks(rect, points, img, col):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    for (x, y) in points:
        cv.circle(img, (x, y), 2, col, -1)

def readPermutations(file, header, footer):
    data = pd.read_csv(file)
    return data.apply(lambda x: header + x + footer).values

if __name__ == "__main__":
    main()