
# Metal Nuts Image Feature Extraction

This project focuses on extracting features from images of "metal nuts" (porcas garra) using a pre-trained Convolutional Neural Network (CNN). The extracted features are organized in a matrix where each row represents an image and each column represents an attribute. The last column of each row indicates the label: `0` for no defects and `1` for defects.

## Dataset Structure

The dataset is divided into the following directories:
- `train/good`: 220 images of good metal nuts.
- `test/bent`: 25 images of bent metal nuts.
- `test/color`: 22 images of color-defective metal nuts.
- `test/flip`: 23 images of flipped metal nuts.
- `test/good`: 22 images of good metal nuts.
- `test/scratch`: 23 images of scratched metal nuts.

## Feature Extraction Process

### Environment Setup

Install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Model Loading

Use VGG16 pre-trained on ImageNet without the top layer.

### Image Preprocessing

There are two preprocessing options:
1. **Normal Preprocessing**: Resize and normalize the images.
2. **Binary Preprocessing**: Convert images to binary using a threshold.

### Feature Extraction

Extract features using the CNN and organize them into a matrix.

### Exporting Features

Save the extracted features to a `.txt` file.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Extracting Features

#### Normal Preprocessing

```python
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load the pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

# Function to extract features and organize into a matrix
def extract_features_to_matrix(directories, labels_map):
    features = []
    for label, directory in directories.items():
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            if os.path.isfile(img_path):
                img_data = preprocess_image(img_path)
                vgg16_feature = model.predict(img_data)
                feature_row = np.append(vgg16_feature.flatten(), labels_map[label])
                features.append(feature_row)
    return np.array(features)

# Define directories and label mapping
directories = {
    'good': 'path_to_train_good_directory',
    'bent': 'path_to_test_bent_directory',
    'color': 'path_to_test_color_directory',
    'flip': 'path_to_test_flip_directory',
    'good_test': 'path_to_test_good_directory',
    'scratch': 'path_to_test_scratch_directory'
}

labels_map = {
    'good': 0,
    'bent': 1,
    'color': 1,
    'flip': 1,
    'good_test': 0,
    'scratch': 1
}

# Extract features and organize into a matrix
features_matrix = extract_features_to_matrix(directories, labels_map)

# Export features to a .txt file
output_file = 'features_matrix.txt'
np.savetxt(output_file, features_matrix, delimiter=',')

print(f"Features and labels have been exported to {output_file}")
```

#### Binary Preprocessing

```python
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

# Function to extract features and organize into a matrix
def extract_features_to_matrix(directories, labels_map):
    features = []
    for label, directory in directories.items():
        for img_name in os.listdir(directory):
            img_path = os.path.join(directory, img_name)
            if os.path.isfile(img_path):
                binary_img = binarize_image(img_path)
                img_data = preprocess_image(binary_img)
                vgg16_feature = model.predict(img_data)
                feature_row = np.append(vgg16_feature.flatten(), labels_map[label])
                features.append(feature_row)
    return np.array(features)

# Define directories and label mapping
directories = {
    'good': 'path_to_train_good_directory',
    'bent': 'path_to_test_bent_directory',
    'color': 'path_to_test_color_directory',
    'flip': 'path_to_test_flip_directory',
    'good_test': 'path_to_test_good_directory',
    'scratch': 'path_to_test_scratch_directory'
}

labels_map = {
    'good': 0,
    'bent': 1,
    'color': 1,
    'flip': 1,
    'good_test': 0,
    'scratch': 1
}

# Extract features and organize into a matrix
features_matrix = extract_features_to_matrix(directories, labels_map)

# Export features to a .txt file
output_file = 'features_matrix.txt'
np.savetxt(output_file, features_matrix, delimiter=',')

print(f"Features and labels have been exported to {output_file}")
```

### Exporting Features to a `.txt` File

The extracted features and labels will be saved to a file named `features_matrix.txt`.

## License

This project is licensed under the MIT License.

## Author

Avando José de L. Campos