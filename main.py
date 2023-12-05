import os, pickle
import numpy as np
from PIL import Image

import streamlit as st
from sklearn.neighbors import NearestNeighbors
from cnn_model import extract_features


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"An error occurred during file upload: {e}")
        return 0


def display_uploaded_file(uploaded_file):
    st.subheader("Uploaded Image")
    display_image = Image.open(uploaded_file)
    display_image = display_image.resize((128, 128))  # Resize to a consistent size
    st.image(display_image)


def find_similar_products(upload_img_path):
    print("\nFinding similar products for the uploaded image..")

    img_features = extract_features(upload_img_path)

    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([img_features])
    return indices


def display_similar_products(indices):
    try:
        st.subheader("Similar product recommendations")
        img_cols = st.columns(5)
        for i, col in enumerate(img_cols):
            with col:
                st.image(filenames[indices[0][i]])
    except Exception as e:
        st.error(f"An error occurred while displaying similar products: {e}")


feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

st.title("Visual-Based Fashion Recommendation System")

# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is None:
    st.info("Please upload an image.")
else:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_uploaded_file(uploaded_file)

        uploaded_img_path = os.path.join("uploads", uploaded_file.name)

        # Display processing message
        processing_msg = st.info("Processing... Finding similar products.")
        indices = find_similar_products(uploaded_img_path)

        if indices is not None:
            # Remove the processing message
            processing_msg.empty()
            display_similar_products(indices)
