import os
import cv2
import mediapipe as mp

####################
use_cpu = True # Disable GPU if needed
if use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
####################

# Inicializa desenhador e modelo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abre webcam
cap = cv2.VideoCapture(0)
window_name = "Deteccao de Mao com MediaPipe"

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.4,
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
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:  # ESC ou 'q'
            break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
