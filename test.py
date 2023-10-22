import pickle
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from cnn_model import extract_features


def recommend(upload_img_path):
    print("\nFinding similar products for the uploaded image..")

    img_features = extract_features(upload_img_path)

    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([img_features])

    print("\nIndices of Recommended Images:", indices)
    return indices


def display_recommendations(indices):
    recommended_images = []

    # Loop through the recommended indices
    for file in indices[0]:
        temp_img = cv2.imread(filenames[file])
        # Resize the image to a consistent size
        temp_img = cv2.resize(temp_img, (256, 256))
        recommended_images.append(temp_img)

    # Stack the images into a 2x2 grid
    grid_image = np.vstack(
        [np.hstack(recommended_images[:2]), np.hstack(recommended_images[2:4])]
    )

    # Display the grid image
    cv2.imshow("Recommended Images", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load pre-computed feature list and corresponding filenames
    feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
    filenames = pickle.load(open("filenames.pkl", "rb"))

    indices = recommend("try/sample/shoe.jpg")
    display_recommendations(indices)
