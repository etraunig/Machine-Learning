import cv2
import mediapipe as mp
import numpy as np

# Inicializa desenhador e modelo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abre webcam
cap = cv2.VideoCapture(0)
window_name = "Pintura com o Dedo (MediaPipe)"

# Lista para armazenar pontos desenhados
draw_points = []

# Cria um canvas preto do mesmo tamanho da tela da webcam
ret, sample_frame = cap.read()
canvas = None
if ret:
    height, width, _ = sample_frame.shape
    canvas = 255 * np.ones_like(sample_frame)  # fundo branco

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Erro ao acessar a câmera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Coordenadas do dedo indicador (landmark 8)
                index_finger_tip = hand_landmarks.landmark[8]
                x = int(index_finger_tip.x * frame.shape[1])
                y = int(index_finger_tip.y * frame.shape[0])
                draw_points.append((x, y))

        # Desenha no canvas com base nos pontos rastreados
        if len(draw_points) > 1:
            for i in range(1, len(draw_points)):
                cv2.line(canvas, draw_points[i - 1], draw_points[i], (0, 0, 255), 4)

        # Sobrepõe o canvas ao frame
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:  # ESC ou 'q' para sair
            break
        elif key == ord('c'):
            draw_points = []
            canvas = 255 * np.ones_like(frame)  # Limpa o canvas

# Libera recursos
cap.release()
cv2.destroyAllWindows()
