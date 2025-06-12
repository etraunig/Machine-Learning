import os
import yaml
import cv2
import math
import mediapipe as mp
from hand_detection import TRIPLETOS

# Função para calcular ângulo (em graus) entre 3 keypoints (em 2D)
def calcular_angulo_2d(a, b, c):
    ab = (a['x'] - b['x'], a['y'] - b['y'])
    cb = (c['x'] - b['x'], c['y'] - b['y'])

    produto_escalar = ab[0]*cb[0] + ab[1]*cb[1]
    norma_ab = math.hypot(*ab)
    norma_cb = math.hypot(*cb)

    if norma_ab == 0 or norma_cb == 0:
        return None

    cos_angulo = produto_escalar / (norma_ab * norma_cb)
    cos_angulo = max(-1.0, min(1.0, cos_angulo))  # Evita que cosseno seja menor q -1 ou maior que 1
    angulo_rad = math.acos(cos_angulo)
    return math.degrees(angulo_rad)


####################
use_cpu = False  # Desabilita GPU se necessário
if use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
####################

# Caminho da imagem de entrada e saída
file_name = "toad"
base_path = "references/"
input_image_path = base_path + "gestures/" + file_name + ".png"
output_image_path = base_path + "annotations/" + file_name + ".jpg"
output_yml_path = base_path + "keypoints/" + file_name + ".yml"
output_angles_path = base_path + "angles/" + file_name + ".yml"

# Inicializa desenhador e modelo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carrega imagem
image = cv2.imread(input_image_path)
if image is None:
    print(f"Erro ao carregar a imagem: {input_image_path}")
    exit()

# Converte para RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Executa detecção com MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.7
) as hands:

    results = hands.process(rgb_image)

    keypoints_data = {}
    angles_data = {}

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks): # itera para cada mao detectada
            hand_name = f"hand_{hand_idx}"
            keypoints = []

            for idx, lm in enumerate(hand_landmarks.landmark): # itera por cada keypoint detectado
                x = lm.x
                y = lm.y
                #z = lm.z
                keypoints.append({
                    "keypoint": idx,
                    "x": x,
                    "y": y
                    #"z": z
                })

            keypoints_data[hand_name] = keypoints

            # Desenha os keypoints na imagem original
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            angles_data[hand_name] = {}
            for a_idx, b_idx, c_idx in TRIPLETOS:
                a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]
                angulo = calcular_angulo_2d(a, b, c)
                key = f"{a_idx}-{b_idx}-{c_idx}"
                angles_data[hand_name][key] = angulo if angulo is not None else "indefinido"

# Salva imagem com anotações
cv2.imwrite(output_image_path, image)
print(f"Imagem salva com anotações em: {output_image_path}")

with open(output_yml_path, "w") as f:
    yaml.dump(keypoints_data, f, sort_keys=False)

print(f"Coordenadas dos keypoints salvas em: {output_yml_path}")

with open(output_angles_path, "w") as f:
    yaml.dump(angles_data, f, sort_keys=False)
print(f"Ângulos salvos em: {output_angles_path}")

