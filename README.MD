# Leitor de Rosto e Objetos com OpenCV e MediaPipe

Este projeto é um aplicativo simples em Python que utiliza a webcam do computador para detectar rostos e objetos em tempo real. Ele faz uso das bibliotecas [OpenCV](https://opencv.org/) e [MediaPipe](https://mediapipe.dev/).

## Como funciona

- **MediaPipe** é utilizado para detectar rostos com alta precisão.
- **OpenCV** é utilizado para detectar objetos usando classificadores Haar Cascade (no exemplo, detecta rostos, mas pode ser adaptado para outros objetos).
- O vídeo da webcam é processado quadro a quadro, desenhando retângulos ao redor dos rostos e objetos detectados.

## Como usar

1. Instale as dependências:
2. Execute o arquivo `app.py`:
3. Uma janela será aberta mostrando o vídeo da webcam com as detecções em tempo real.
4. Pressione `ESC` para sair.

## Observações

- Para detectar outros objetos além de rostos, basta trocar o classificador Haar Cascade utilizado no código.
- Para detectar carros, por exemplo, é necessário baixar o arquivo `haarcascade_car.xml` e ajustar o código.