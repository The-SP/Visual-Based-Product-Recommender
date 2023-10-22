import os, sys, pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# Load pre-trained ResNet50 model
resnet_model = ResNet50(
    weights="imagenet",  # Initialize with pre-trained weights
    include_top=False,  # Exclude fully connected layers at the top
    input_shape=(224, 224, 3),  # Set input shape to (224, 224, 3)
)
# Freeze the pre-trained model (make it non-trainable). This means that the weights of these layers will not be updated during the training process.
resnet_model.trainable = False

# Create a new Sequential model
model = tf.keras.Sequential(
    [
        resnet_model,  # Add the pre-trained ResNet50 model
        GlobalMaxPooling2D(),  # Add Global Max Pooling layer
    ]
)
# print(model.summary())


# Extract features from image using a pre-trained model
def extract_features(img_path, model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Use the model to make predications and normalize the result
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result


def extract_features_batch(img_paths, model, batch_size=16):
    feature_list = []
    total_images = len(img_paths)

    start_index = 0
    if os.path.exists("embeddings.pkl"):
        feature_list = pickle.load(open("embeddings.pkl", "rb"))
        start_index = len(feature_list)
        print(f"Resuming from {start_index}/{total_images} images")

    try:
        for i in range(start_index, total_images, batch_size):
            # Load and preprocess the images
            batch_paths = img_paths[i : i + batch_size]
            images = [
                image.load_img(path, target_size=(224, 224)) for path in batch_paths
            ]
            img_arrays = [image.img_to_array(img) for img in images]
            expanded_img_arrays = np.array(
                [np.expand_dims(arr, axis=0) for arr in img_arrays]
            )
            preprocessed_imgs = [
                preprocess_input(expanded_img_array)
                for expanded_img_array in expanded_img_arrays
            ]

            # Use the model to make predictions and normalize the result
            results = [
                model.predict(preprocessed_img, verbose=0).flatten()
                for preprocessed_img in preprocessed_imgs
            ]
            normalized_results = [result / np.linalg.norm(result) for result in results]

            feature_list.extend(normalized_results)

            print(f"Processed {i + batch_size}/{total_images} images")

            # Save features every 10 batches
            if (i % (batch_size * 10)) < batch_size and i > 0:
                pickle.dump(feature_list, open("embeddings.pkl", "wb"))
                print(f"\nSaved features up to image {i + batch_size}/{total_images}\n")

        # Save the extracted feature to pickle files
        pickle.dump(feature_list, open("embeddings.pkl", "wb"))

    except KeyboardInterrupt:
        pickle.dump(feature_list, open("embeddings.pkl", "wb"))
        print(
            f"\nProcessing interrupted. Saved current progress up to image {i}/{total_images}."
        )
        sys.exit()


if __name__ == "__main__":
    # Create a list of file paths for images in the 'images' directory
    filenames = [
        os.path.join("data/images", file) for file in os.listdir("data/images")
    ]
    # Save the file names to pickle files
    # pickle.dump(filenames, open("filenames.pkl", "wb"))

    # Loop through the file paths and extract features from each image
    extract_features_batch(filenames, model)
