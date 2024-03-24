import cv2
import numpy as np


def filtro_passa_baixa_media(imagem, tamanho_kernel):
    return cv2.blur(imagem, (tamanho_kernel, tamanho_kernel))


# Exemplo de uso:
imagem = cv2.imread('imagem.png', 0)  # Carregar imagem em escala de cinza
imagem_filtrada_media = filtro_passa_baixa_media(imagem, 5)  # Aplicar filtro com kernel 5x5
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Filtrada (Média)', imagem_filtrada_media)
cv2.waitKey(0)
cv2.destroyAllWindows()
