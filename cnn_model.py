import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


def get_model():
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

    print(model.summary())
    return model


# Extract features from image using a pre-trained model
def extract_features(img_path, model=get_model()):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    # Use the model to make predications and normalize the result
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)

    return normalized_result
