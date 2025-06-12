# Detector da Tecnica das Dez Sombras
Group: Bianca Zuchinali, Eduardo Traunig, Erick Machado, Ingrid Carolina, Thiago Zilberknop

A Técnica das Dez Sombras é uma poderosa técnica hereditária do Clã Zenin no anime e mangá Jujutsu Kaisen. Ela é usada principalmente por Megumi Fushiguro, um dos personagens principais. Essa técnica utiliza sinais com as mãos para invocar diferentes shikigami a partir da sombra do usuário. Cada shikigami possui habilidades únicas e funções específicas em combate.

Shikigami detectados:

Cães Divinos – Lobo

Nue – Pássaro

Sapo – Rã (o mediapipe não consegue estimar direito os pontos referentes as pontas dos dedos nesse gesto, dificultando a deteccção deste gesto)

# Como compilar e rodar:

- Instale Python
- Crie a venv: `python3 -m venv .`
- Ative a venv: `source bin/activate`
- Instale os requisitos com `pip install -r requirements.txt`
- Rode o programa com `python hand_detection.py`

# Como registrar novas poses:
- Coloque a imagem da mao no folder references/gestures com o nome do gesto como nome do arquivo
- Rodar o script detect_from_image.py passando como argumento o nome do gesto (sem a extensao do arquivo)
- Após rodar o script, ele irá gerar os dados necessários para que a detecção de poses funcione. Rode o arquivo hand_detection.py e veja o gesto ser detectado!

