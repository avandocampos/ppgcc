import numpy as np
from PIL import Image


def multi_thresholding(image, thresholds):
    thresholds = sorted(thresholds + [0, 255])
    for i in range(1, len(thresholds)):
        image[(image >= thresholds[i-1]) & (image < thresholds[i])] = i - 1
    return (image * (255 // (len(thresholds) - 1))).astype('uint8')


if __name__ == '__main__':

    image = Image.open('images/sample.jpg').convert('L')  # Load image in grayscale
    image = np.array(image)
    multi_thresholded_image = multi_thresholding(image, [64, 128, 192])  # Apply multi-thresholding
    Image.fromarray(multi_thresholded_image).save('images/multi_thresholded_image.jpg')  # Save multi-thresholded image