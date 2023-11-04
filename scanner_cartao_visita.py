import cv2
import numpy as np

def find_document_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    document_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            document_contour = contour

    return document_contour

def preprocess(image):
    height, width, _ = image.shape
    max_dimension = max(height, width)
    scale = 512.0 / max_dimension
    new_height, new_width = int(height * scale), int(width * scale)
    image = cv2.resize(image, (new_width, new_height))
    image = image[1:-1, 1:-1]
    return image

def scan_card(image_path):
    image = cv2.imread(image_path)
    preprocessed_image = preprocess(image)
    contour = find_document_contour(preprocessed_image)

    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        scanned_image = preprocessed_image[y:y+h, x:x+w]
        cv2.imwrite('scanned_card.jpg', scanned_image)
        print('Scanned card image has been saved as scanned_card.jpg')
    else:
        print('Failed to find the card in the image')

if __name__ == '__main__':
    scan_card('path/to/your/image.jpg')