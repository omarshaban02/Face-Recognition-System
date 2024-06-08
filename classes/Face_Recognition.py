import os
import numpy as np
import cv2

def load_pca_model(load_folder):
    centered_data = np.load(os.path.join(load_folder, 'centered_data.npy'))
    eigenfaces = np.load(os.path.join(load_folder, 'eigenfaces.npy'))
    mean_image = np.load(os.path.join(load_folder, 'mean_image.npy'))

    return centered_data, eigenfaces, mean_image


def transform_data(centered_data, selected_eigenvectors):
    transformed_data = np.dot(selected_eigenvectors.T, centered_data)

    return transformed_data


def predict_pca(test_image, eigenfaces, mean_image, transformed_data):
    # Resize the test image to match the target size
    test_image_resized = cv2.resize(test_image, (64, 64))

    # Flatten the resized test image
    test_image_vector = test_image_resized.flatten()

    # Center the test image by subtracting the mean image
    centered_test_image = test_image_vector - mean_image

    # Project the centered test image onto the eigenfaces
    test_image_transformed = np.dot(eigenfaces.T, centered_test_image)

    # Calculate Euclidean distances between the transformed test image and training images
    distances = np.linalg.norm(
        transformed_data - test_image_transformed[:, np.newaxis], axis=0)
    # print(transformed_data.shape)

    # Find the index of the closest match
    closest_index = np.argmin(distances)
    # print("closest_index:", closest_index)

    predicted_subject = closest_index // 10 + 1
    print("predicted_subject", predicted_subject)

    return predicted_subject
