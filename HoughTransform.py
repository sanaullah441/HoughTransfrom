import cv2
import numpy as np
from skimage import io

# read image from disk
img = cv2.imread('sanaullah_shakil.png')
# convert RGB to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# find edges using canny edge detector
cannyEdges = cv2.Canny(gray, 100, 150, apertureSize=3)
# find lines in image
lines = cv2.HoughLines(cannyEdges, 1, np.pi / 180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255,), 2)

# show result
cv2.imshow('Canny Edge Detection', cannyEdges)
cv2.imshow('Hough Transform lines', img)

# save result
io.imsave('result/Canny detected edges.png', cannyEdges)
io.imsave('result/hough lines.png', img)

# harris Corner detector
gray = np.float32(gray)
harrisCorner = cv2.cornerHarris(gray, 4, 3, 0, 0.04)

# result is dilated for marking the corners,not important
dst = cv2.dilate(harrisCorner, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# show result
cv2.imshow('Hough Transform lines & harris corners', img)
# save result in disk
io.imsave('result/Harris Corner detection & HT lines.png', img)

# waiting for key
k = cv2.waitKey(0)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
