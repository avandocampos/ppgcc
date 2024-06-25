import numpy as np
import os
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load the pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to binarize the image
def binarize_image(img_path, threshold=127):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels
    return binary_img

# Function to preprocess the image
def preprocess_image(img):
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# Function to extract features and organize them into a dictionary
def extract_features_to_matrix(directories, labels_map, labels_good_or_not):
    features_dict = {label: [] for label in labels_map.values()}
    for label, directory in directories.items():
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            if os.path.isfile(img_path):
                binary_img = binarize_image(img_path)
                img_data = preprocess_image(binary_img)
                vgg16_feature = model.predict(img_data)
                feature_row = np.append(vgg16_feature.flatten(), labels_good_or_not[label])
                # feature_row = np.append(feature_row, labels_good_or_not[label])
                features_dict[labels_map[label]].append(feature_row)
    return features_dict

# Define directories and label mapping
directories = {
    'good': 'metal_nut/train/good',
    'bent': 'metal_nut/test/bent',
    'color': 'metal_nut/test/color',
    'flip': 'metal_nut/test/flip',
    'good_test': 'metal_nut/test/good',
    'scratch': 'metal_nut/test/scratch'
}

labels_map = {
    'good': 0,
    'bent': 1,
    'color': 2,
    'flip': 3,
    'good_test': 4,
    'scratch': 5
}

labels_good_or_not = {
    'good': 0,
    'bent': 1,
    'color': 1,
    'flip': 1,
    'good_test': 0,
    'scratch': 1
}

# Extract features and organize into a dictionary
features_dict = extract_features_to_matrix(directories, labels_map, labels_good_or_not)

# Export features to separate .txt files per class
for label, features in features_dict.items():
    output_file = f'features_matrix_class_{label}.txt'
    np.savetxt(output_file, features, delimiter=',')

print("Features and labels have been exported to separate files per class.")
