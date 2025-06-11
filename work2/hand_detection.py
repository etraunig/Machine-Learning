import os
import cv2
import mediapipe as mp
import numpy as np
import yaml
import math

####################
use_cpu = True
if use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
####################

# Define os mesmos tripletos do arquivo detect_from_image.py
TRIPLETOS = [
    (1, 2, 3), (2, 3, 4),     # Polegar completo até a ponta
    (5, 6, 7), (6, 7, 8),     # Indicador completo até a ponta
    (9, 10, 11), (10, 11, 12), # Médio completo até a ponta
    (13, 14, 15), (14, 15, 16), # Anelar completo até a ponta
    (17, 18, 19), (18, 19, 20)  # Mínimo completo até a ponta
]

# === Funções de gesto ===

def carregar_angulos_yaml(caminho):
    with open(caminho, "r") as f:
        data = yaml.safe_load(f)
    return data["hand_0"]

def carregar_angulos_de_diretorio(diretorio):
    angulos_por_arquivo = {}
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.endswith(".yml"):
            caminho = os.path.join(diretorio, nome_arquivo)
            with open(caminho, "r") as f:
                dados = yaml.safe_load(f)
                nome_base = os.path.splitext(nome_arquivo)[0]
                angulos_por_arquivo[nome_base] = {
                    "hand_0": dados.get("hand_0", {}),
                    "hand_1": dados.get("hand_1", {})
                }
    return angulos_por_arquivo

def calcular_angulo_2d(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    produto_escalar = ab[0]*cb[0] + ab[1]*cb[1]
    norma_ab = math.hypot(*ab)
    norma_cb = math.hypot(*cb)
    if norma_ab == 0 or norma_cb == 0:
        return None
    cos_angulo = produto_escalar / (norma_ab * norma_cb)
    cos_angulo = max(-1.0, min(1.0, cos_angulo))
    angulo_rad = math.acos(cos_angulo)
    return math.degrees(angulo_rad)

def calcular_angulos_frame(landmarks):
    angulos = {}
    for a_idx, b_idx, c_idx in TRIPLETOS:
        p_a = landmarks[a_idx]
        p_b = landmarks[b_idx]
        p_c = landmarks[c_idx]
        angulo = calcular_angulo_2d(p_a, p_b, p_c)
        if angulo is not None:
            angulos[f"{a_idx}-{b_idx}-{c_idx}"] = angulo
    return angulos

def comparar_angulos(ang_atual, ang_salvo, threshold=3.0):
    valores_comparados = []
    for chave, angulo_salvo in ang_salvo.items():
        if chave in ang_atual and isinstance(angulo_salvo, (int, float)):
            valores_comparados.append((ang_atual[chave] - angulo_salvo) ** 2)

    if not valores_comparados:
        return False

    rmse = np.sqrt(np.mean(valores_comparados))
    return rmse < threshold


def salvar_imagem_e_gesto(frame, angulos, base_pasta="references", nome="gesto_salvo"):
    pasta_img = os.path.join(base_pasta, "gestures")
    pasta_angles = os.path.join(base_pasta, "angles")
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(pasta_angles, exist_ok=True)

    # Salva imagem
    caminho_img = os.path.join(pasta_img, f"{nome}.jpg")
    cv2.imwrite(caminho_img, frame)

    # Salva ângulos YAML
    caminho_yml = os.path.join(pasta_angles, f"{nome}.yml")
    with open(caminho_yml, "w") as f:
        yaml.dump({"hand_0": angulos}, f, sort_keys=False)

    print(f"Imagem salva em: {caminho_img}")
    print(f"Ângulos salvos em: {caminho_yml}")


if __name__ == "__main__":

    # === Inicialização do MediaPipe ===

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Carrega os ângulos do gesto salvo
    diretorio_angulos = "references/angles"
    angulos_salvos_dict = carregar_angulos_de_diretorio(diretorio_angulos)

    # Abre webcam
    cap = cv2.VideoCapture(0)
    window_name = "Deteccao de Mao com MediaPipe"

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Erro ao acessar a câmera.")
                break

            frame = cv2.flip(frame, 1)
            frame_original = frame.copy()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            angulos_atuais = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # Extrair pontos normalizados
                    pontos = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    angulos_atuais = calcular_angulos_frame(pontos)

                    for nome_gesto, gesto_data in angulos_salvos_dict.items():
                        if comparar_angulos(angulos_atuais, gesto_data["hand_0"]):
                            cv2.putText(frame, f"GESTO: {nome_gesto.upper()}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            break  # Interrompe no primeiro gesto detectado

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]:  # ESC ou q
                break
            elif key == 32 and angulos_atuais is not None:  # Barra de espaço
                salvar_imagem_e_gesto(frame_original, angulos_atuais, nome="nue")

    cap.release()
    cv2.destroyAllWindows()

