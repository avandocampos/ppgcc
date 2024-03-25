import cv2


def filtro_passa_baixa_media(imagem, tamanho_kernel):
    altura, largura = imagem.shape
    imagem_filtrada = imagem.copy().astype(float)
    offset = tamanho_kernel // 2

    for y in range(offset, altura - offset):
        for x in range(offset, largura - offset):
            sub_imagem = imagem[y - offset:y + offset + 1, x - offset:x + offset + 1]
            imagem_filtrada[y, x] = (sub_imagem.sum() / (tamanho_kernel ** 2)).astype('uint8')

    cv2.normalize(imagem_filtrada, imagem_filtrada, 0, 255, cv2.NORM_MINMAX)

    return imagem_filtrada.astype('uint8')


if __name__ == '__main__':

    imagem = cv2.imread('imagem.jpeg', 0)  # Carregar imagem em escala de cinza
    imagem_filtrada_media = filtro_passa_baixa_media(imagem, 5)  # Aplicar filtro com kernel 5x5
    cv2.imwrite('imagem_filtrada_media.jpeg', imagem_filtrada_media)
