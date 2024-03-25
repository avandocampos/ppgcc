import cv2


def low_pass_filter_average(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy().astype(float)
    offset = kernel_size // 2

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            sub_image = image[y - offset:y + offset + 1, x - offset:x + offset + 1]
            filtered_image[y, x] = (sub_image.sum() / (kernel_size ** 2)).astype('uint8')

    cv2.normalize(filtered_image, filtered_image, 0, 255, cv2.NORM_MINMAX)

    return filtered_image.astype('uint8')


if __name__ == '__main__':

    image = cv2.imread('imagem.jpeg', 0)  # Load image in grayscale
    filtered_image_average = low_pass_filter_average(image, 5)  # Apply filter with 5x5 kernel
    cv2.imwrite('filtered_image_average.jpeg', filtered_image_average)
