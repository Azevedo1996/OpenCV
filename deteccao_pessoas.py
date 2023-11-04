import cv2
import numpy as np

# Inicializa contador de pessoas
contador_pessoas = 0

# Função para tratar e detectar os blobs (objetos) no frame
def detec_blobs(frame):
    global contador_pessoas

    # Converte a cor do frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduz ruído no frame escalado de cinza
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Aplica o algoritmo de limiarização
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Aplica o algoritmo de detecção de contornos (edges)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Conta o número de contornos encontrados (que podem representar pessoas)
    contador_pessoas = len(contours)

    # Desenha os contornos detectados no frame original
    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)

    # Retorna o frame original com os contornos desenhados
    return frame

# Função para contar o número de pessoas no vídeo
def contagem_pessoas(video):
    global contador_pessoas

    # Inicializa a captura de vídeo a partir do arquivo fornecido
    cap = cv2.VideoCapture(video)

    # Define um limite para a taxa de quadros por segundo
    fps_limite = 30

    # Conta a duração total do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calcula a quantidade de quadros a serem pulados para alcançar o limite de fps desejado
    skip_frames = round(total_frames / fps_limite)

    # Inicializa a variável para controlar o número de quadros lidos
    frame_atual = 0

    # Lê o primeiro frame do vídeo
    ret, frame = cap.read()

    # Executa a função de detecção de blobs apenas em quadros selecionados
    while ret:
        frame_atual += 1
        if frame_atual % skip_frames == 0:
            frame = detec_blobs(frame)
            cv2.putText(frame, "Pessoas: {}".format(contador_pessoas), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Contagem de Pessoas', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    # Libera os recursos utilizados e fecha as janelas abertas
    cap.release()
    cv2.destroyAllWindows()

# Utilize a função "contagem_pessoas" passando o caminho do arquivo de vídeo como argumento
contagem_pessoas('caminho/para/seu/video.mp4')
