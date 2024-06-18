import numpy as np

def erosion(image, kernel):
    pad_height, pad_width = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    eroded_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            # Verificar se a região corresponde ao kernel
            if np.array_equal(region, kernel * 255):
                eroded_image[i, j] = 255  # Valor do pixel branco

    return eroded_image

import numpy as np

def dilation(image, kernel):
    # Calcular o padding necessário para a imagem
    pad_height, pad_width = kernel.shape[0] // 2, kernel.shape[1] // 2
    # Pad a imagem com zeros nas bordas
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # Criar uma imagem de saída para armazenar o resultado da dilatação
    dilated_image = np.zeros_like(image)

    # Percorrer cada pixel da imagem original
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Verificar se alguma posição da vizinhança (definida pelo kernel) contém um pixel branco
            if np.any(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel):
                dilated_image[i, j] = 255  # Definir o pixel como branco na imagem dilatada

    return dilated_image
