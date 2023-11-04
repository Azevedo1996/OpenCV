import cv2
import numpy as np

def empty(a):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)

cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",179,179,empty)
cv2.createTrackbar("SAT Min","HSV",0,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

cv2.setTrackbarPos("HUE Min","HSV",0)
cv2.setTrackbarPos("HUE Max","HSV",179)
cv2.setTrackbarPos("SAT Min","HSV",0)
cv2.setTrackbarPos("SAT Max","HSV",255)
cv2.setTrackbarPos("VALUE Min","HSV",0)
cv2.setTrackbarPos("VALUE Max","HSV",255)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max","HSV")
    s_min = cv2.getTrackbarPos("SAT Min","HSV")
    s_max = cv2.getTrackbarPos("SAT Max","HSV")
    v_min = cv2.getTrackbarPos("VALUE Min","HSV")
    v_max = cv2.getTrackbarPos("VALUE Max","HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    cv2.imshow('HSV', hsv)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()