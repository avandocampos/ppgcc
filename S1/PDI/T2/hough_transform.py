import numpy as np
import cv2
from numba import njit, prange


def hough_lines(image, rho_resolution=1, theta_resolution=np.pi/180, threshold=100):
    rows, cols = image.shape
    max_rho = int(np.hypot(rows, cols))
    accumulator = np.zeros((2 * max_rho, int(np.pi / theta_resolution)))

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255:
                for theta in range(accumulator.shape[1]):
                    rho = int(i * np.cos(theta * theta_resolution) + j * np.sin(theta * theta_resolution))
                    accumulator[rho + max_rho, theta] += 1

    lines = []
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] >= threshold:
                lines.append((rho - max_rho, theta * theta_resolution))

    return lines


def hough_circles(image, r_min, r_max, threshold):
    rows, cols = image.shape
    # Inicializando o acumulador
    accumulator = np.zeros((rows, cols, r_max - r_min))
    
    # Percorrendo cada pixel da imagem
    for x in range(rows):
        for y in range(cols):
            if image[x, y] == 255:
                for r in range(r_min, r_max):
                    for theta in range(0, 360):
                        a = int(x - r * np.cos(theta * np.pi / 180))
                        b = int(y - r * np.sin(theta * np.pi / 180))
                        if 0 <= a < rows and 0 <= b < cols:
                            accumulator[a, b, r - r_min] += 1

    circles = []
    # Procurando valores no acumulador acima do limiar
    for r in range(r_max - r_min):
        for x in range(rows):
            for y in range(cols):
                if accumulator[x, y, r] >= threshold:
                    circles.append((x, y, r + r_min))

    return circles


@njit(parallel=True)
def hough_circles_optimized(image, r_min, r_max, threshold):
    rows, cols = image.shape
    accumulator = np.zeros((rows, cols, r_max - r_min), dtype=np.int32)

    for x in prange(rows):
        for y in range(cols):
            if image[x, y] == 255:
                for r in range(r_min, r_max):
                    for theta in range(0, 360, 3):  # Usar um incremento maior para theta diminui o tempo de processamento
                        a = int(x - r * np.cos(theta * np.pi / 180))
                        b = int(y - r * np.sin(theta * np.pi / 180))
                        if 0 <= a < rows and 0 <= b < cols:
                            accumulator[a, b, r - r_min] += 1

    circles = []
    for r in range(r_max - r_min):
        for x in range(rows):
            for y in range(cols):
                if accumulator[x, y, r] >= threshold:
                    circles.append((x, y, r + r_min))

    return circles