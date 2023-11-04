import cv2
import numpy as np

# Capture a reference image from camera
def capture_reference_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Reference Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame

# Perform color detection using a given reference image
def perform_color_detection(frame, color, threshold):
    # Convert frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of color to detect
    lower_color = np.array(color, dtype=np.uint8) - threshold
    upper_color = np.array(color, dtype=np.uint8) + threshold

    # Threshold the HSV image to get only color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Perform the bitwise AND operation between frame and mask
    res = cv2.bitwise_and(frame, frame, mask=mask)

    return res, mask

# Initialize HSV color for color detection
def init_color():
    print("Enter the color in (Blue, Green, Red) format: ")
    color = tuple(map(int, input().split()))

    print("Enter the threshold for color detection: ")
    threshold = int(input())

    return color, threshold

def main():
    color, threshold = init_color()
    ref_image = capture_reference_image()
    res, mask = perform_color_detection(ref_image, color, threshold)

    cv2.imshow('Result', res)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()