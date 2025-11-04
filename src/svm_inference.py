import joblib
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize

# Load model + scaler
svm_model = joblib.load("models/bin_svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def extract_hog_single(image, target_size=(128,128)):
    img = np.array(image)

    if img.ndim == 3:
        img = rgb2gray(img)

    img_resized = resize(img, target_size, anti_aliasing=True)

    hog_features = hog(
        img_resized,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        feature_vector=True
    )

    mean_intensity = np.mean(img_resized)
    std_intensity = np.std(img_resized)

    feature_vector = np.concatenate([hog_features, [mean_intensity, std_intensity]])
    return feature_vector.reshape(1, -1)


def predict_svm(image):
    features = extract_hog_single(image)
    features_scaled = scaler.transform(features)

    pred = svm_model.predict(features_scaled)[0]
    prob = svm_model.predict_proba(features_scaled)[0][1]

    return pred, prob