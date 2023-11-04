import cv2

# Carrega o modelo HOG (Histogram of Oriented Gradients)
pedestrian_detector = cv2.HOGDescriptor()
pedestrian_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Abre a câmera e lê um quadro
cap = cv2.VideoCapture(0)
_, frame = cap.read()

# Converte o frame para escala de cinza
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detecta pedestres no frame escalar de cinza
bbox, _ = pedestrian_detector.detectMultiScale(gray, winStride=(4, 4),
                                                    padding=(8, 8), scale=1.05)

# Desenha retângulos verdes em torno dos pedestres detectados
for i in range(len(bbox)):
    cv2.rectangle(frame, (bbox[i][0], bbox[i][1]),
                 (bbox[i][0] + bbox[i][2], bbox[i][1] + bbox[i][3]),
                 (0, 255, 0), 2)

# Exibe o frame com os retângulos verdes desenhados
cv2.imshow('Pedestrian Detector', frame)

# Libera a câmera e fecha todas as janelas abertas
cap.release()
cv2.destroyAllWindows()