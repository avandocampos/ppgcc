import numpy as np
from PIL import Image


def thresholding(image, threshold):
    return ((image > threshold) * 255).astype('uint8')


if __name__ == '__main__':

    image = Image.open('images/sample.jpg').convert('L')  # Load image in grayscale
    image = np.array(image)
    thresholded_image = thresholding(image, 128)  # Apply thresholding
    Image.fromarray(thresholded_image).save('images/thresholded_image.jpg')  # Save thresholded image