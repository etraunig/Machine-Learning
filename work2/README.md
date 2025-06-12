# Detector da Tecnica das Dez Sombras (+ 2 expansões de domínio)
Group: Bianca Zuchinali, Eduardo Traunig, Erick Machado, Ingrid Carolina, Thiago Zilberknop

A Técnica das Dez Sombras é uma poderosa técnica hereditária do Clã Zenin no anime e mangá Jujutsu Kaisen. Ela é usada principalmente por Megumi Fushiguro, um dos personagens principais. Essa técnica utiliza sinais com as mãos para invocar diferentes shikigami a partir da sombra do usuário. Cada shikigami possui habilidades únicas e funções específicas em combate.

Shikigami detectados:
- Divine Dogs – Lobo
- Nue – Pássaro
- Toad – Rã (detecção mais díficil que os outros)

Expansões de Domínio:
- Idle Death Gamble -> Kinji Hakari
- Unlimited Void -> Satoru Gojo

# Como compilar e rodar:

- Instale Python
- Crie a venv: `python3 -m venv .`
- Ative a venv: `source bin/activate`
- Instale os requisitos com `pip install -r requirements.txt`
- Rode o programa com `python hand_detection.py`

# Como registrar novas poses:
- Coloque a imagem da mão na pasta references/gestures/ com o nome do arquivo sendo o nome do gesto.
- Rodar o script `detect_from_image.py` passando como argumento o nome do arquivo com a extensao (o código procura na pasta references/gestures/).
- Após rodar o script, ele irá gerar os dados necessários para que a detecção de poses funcione. Rode o arquivo `hand_detection.py` e veja os gestos serem detectados (somente os gestos que passaram pelo `detect_from_image.py` são detectados.
- (OPCIONAL) O script `hand_detection.py` pode ser usado para registrar gestos também, ao precionar espaço ele inicia um countdown para salvar a imagem da webcam que ele está vendo na pasta correta, basta apenas renomear o arquivo e usar o `detect_from_image.py` para adicionar o mesmo a lista de detecção.

