import cv2
import numpy as np

def detect_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            return True

    return False

def main():
    image_path = 'path/to/your/image.jpg'
    image = cv2.imread(image_path)

    if detect_license_plate(image):
        print('License plate detected!')
    else:
        print('No license plate detected.')

if __name__ == '__main__':
    main()