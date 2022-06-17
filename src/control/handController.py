import time

import autopy
import cv2
import mouse
import numpy as np
from pynput.keyboard import Controller

from src.detection import HandTrackingModule as htm

w_cam, h_cam = 640, 480
frame_reduction = 100
cap = cv2.VideoCapture(0)
keyboard = Controller()
smoothening = 7

if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(3, w_cam)
cap.set(4, h_cam)
pTime = 0
prev_location_x, prev_location_y = 0, 0
current_location_x, current_location_y = 0, 0

detector = htm.handDetector(maxHands=1)
width_screen, height_screen = autopy.screen.size()
cute_cats_checker = 0

while True:
    # find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # use index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)little cats images

        # check if the fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frame_reduction, frame_reduction), (w_cam - frame_reduction, h_cam - frame_reduction),
                      (255, 0, 255), 2)

        # moving mode: index finger and convert coordinates
        if fingers == [0, 1, 0, 0, 0]:
            x3 = np.interp(x1, (frame_reduction, w_cam - frame_reduction), (0, width_screen))
            y3 = np.interp(y1, (frame_reduction, h_cam - frame_reduction), (0, height_screen))
            cute_cats_checker = 0

            # smoothen values
            current_location_x = prev_location_x + (x3 - prev_location_x) / smoothening
            current_location_y = prev_location_y + (y3 - prev_location_y) / smoothening

            # mouse movement
            autopy.mouse.move(width_screen - current_location_x, current_location_y)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            prev_location_x, prev_location_y = current_location_x, current_location_y

        # click mode: both index and middle are up
        elif fingers == [0, 1, 1, 0, 0]:
            # calculate distance
            length, img, information = detector.findDistance(8, 12, img)
            # print(length)
            # click mouse if distance if short
            if length < 40:
                cv2.circle(img, (information[4], information[5]), 10, (255, 255, 0), cv2.FILLED)
                autopy.mouse.click(autopy.mouse.Button.LEFT)

        # right click
        elif fingers == [0, 0, 0, 0, 1]:
            cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
            autopy.mouse.click(autopy.mouse.Button.RIGHT)

        # cute cats
        elif fingers == [1, 0, 0, 0, 1]:
            if cute_cats_checker == 0:
                keyboard.type('cute little cats images')
            cute_cats_checker = 1

        # scroll up and scroll down
        elif fingers == [0, 1, 0, 0, 1]:
            # calculate distance
            length, img, information = detector.findDistance(8, 20, img)
            # print(length)
            # click mouse if distance if short
            if length > 50:
                cv2.circle(img, (information[4], information[5]), 10, (255, 255, 0), cv2.FILLED)
                mouse.wheel(1)
                time.sleep(0.3)
            else:
                cv2.circle(img, (information[4], information[5]), 10, (255, 255, 0), cv2.FILLED)
                mouse.wheel(-1)
                time.sleep(0.3)

    # frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (138, 43, 226), 3)
    # display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
