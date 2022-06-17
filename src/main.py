import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

w_cam, h_cam = 640, 480

cap = cv2.VideoCapture(1)

# if not cap.isOpened():
#     raise IOError("Cannot open webcam")
# cap.set(3, w_cam)
# cap.set(4, h_cam)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
