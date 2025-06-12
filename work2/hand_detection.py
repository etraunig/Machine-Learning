import os
import cv2
import mediapipe as mp
import numpy as np
import yaml
import math
import time

contagem_em_andamento = False
tempo_inicio = 0
tempo_espera = 5 # segundos

####################
use_cpu = False
if use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"
####################

TRIPLETOS = [
    (1, 2, 3), (2, 3, 4),     # Polegar completo até a ponta
    (5, 6, 7), (6, 7, 8),     # Indicador completo até a ponta
    (9, 10, 11), (10, 11, 12), # Médio completo até a ponta
    (13, 14, 15), (14, 15, 16), # Anelar completo até a ponta
    (17, 18, 19), (18, 19, 20)  # Mínimo completo até a ponta
]

# === Funções de gesto ===

def carregar_angulos_de_diretorio(diretorio):
    angulos_por_arquivo = {}
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.endswith(".yml"):
            caminho = os.path.join(diretorio, nome_arquivo)
            with open(caminho, "r") as f:
                dados = yaml.safe_load(f)
                nome_base = os.path.splitext(nome_arquivo)[0]
                angulos_por_arquivo[nome_base] = dados  # Agora com chaves 'Right' e 'Left'
                # número de mãos presentes no gesto
                angulos_por_arquivo[nome_base]["num_hands"] = sum(
                    1 for k in ["Right", "Left"] if dados.get(k)
                )
    return angulos_por_arquivo

def calcular_angulo_2d(a, b, c):    # calcula o angulo entre AB e CB
    # calcula vetores ab e cb
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    # calcula produto escalar: u . v = x1 * x2 + y1 * y2
    produto_escalar = ab[0]*cb[0] + ab[1]*cb[1]
    # norma (magnitude) dos vetores
    norma_ab = math.hypot(*ab)
    norma_cb = math.hypot(*cb)
    if norma_ab == 0 or norma_cb == 0:
        return None
    # cos(x) = (u . v) / (|u| * |v|)      
    cos_angulo = produto_escalar / (norma_ab * norma_cb)
    cos_angulo = max(-1.0, min(1.0, cos_angulo)) # evita erros de arredondamento onde o cos pode ser maior que 1 ou menor que -1
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

def comparar_angulos(ang_atual, ang_salvo, threshold=3.5):
    valores_comparados = []
    for chave, angulo_salvo in ang_salvo.items():
        if chave in ang_atual and isinstance(angulo_salvo, (int, float)):
            # calcula erro quadratico entre os angulos
            valores_comparados.append((ang_atual[chave] - angulo_salvo) ** 2)
    if not valores_comparados:
        return False
    # Root Mean Squared Error -> raiz da media dos erros quadraticos
    rmse = np.sqrt(np.mean(valores_comparados))
    return rmse < threshold


def salvar_imagem(frame, base_pasta="references", nome="gesto_salvo"):
    pasta_img = os.path.join(base_pasta, "gestures")
    pasta_angles = os.path.join(base_pasta, "angles")
    os.makedirs(pasta_img, exist_ok=True)
    os.makedirs(pasta_angles, exist_ok=True)

    # Salva imagem
    caminho_img = os.path.join(pasta_img, f"{nome}.jpg")
    cv2.imwrite(caminho_img, frame)

    print(f"Imagem salva em: {caminho_img}")


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

            if results.multi_hand_landmarks and results.multi_handedness:  
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Obtém se é 'Right' ou 'Left'
                    label = handedness.classification[0].label
                    
                    # Desenha landmarks e Label
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(frame, label, (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Extrair pontos normalizados
                    pontos = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    angulos_atuais = calcular_angulos_frame(pontos)
                    
                # Compara com os gestos salvos
                for nome_gesto, dados_angulo in angulos_salvos_dict.items():
                    # Verifica se tem mãos suficientes para o gesto
                    num_maos_necessarias = dados_angulo.get('num_hands', 1)
                    if len(results.multi_hand_landmarks) != num_maos_necessarias:
                        continue

                    # Se o YML não tiver ângulos para essa mão, pula
                    if label not in dados_angulo:
                        continue

                    if comparar_angulos(angulos_atuais, dados_angulo[label]):
                        cv2.putText(frame, f"GESTO: {nome_gesto.upper()}", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        break
                
                if contagem_em_andamento:
                    tempo_decorrido = time.time() - tempo_inicio
                    tempo_restante = max(0, int(tempo_espera - tempo_decorrido))
                    
                    # Mostra o countdown na tela
                    cv2.putText(frame, f"{tempo_restante}s", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    
                    if tempo_decorrido >= tempo_espera:
                        contagem_em_andamento = False
                        tempo_inicio = 0
                        salvar_imagem(frame_original, nome="novo")
                        print("Gesto salvo com sucesso!")

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q')]:  # ESC ou q
                break
            elif key == 32 and angulos_atuais is not None:  # Barra de espaço 
                contagem_em_andamento = True
                tempo_inicio = time.time()

    cap.release()
    cv2.destroyAllWindows()

