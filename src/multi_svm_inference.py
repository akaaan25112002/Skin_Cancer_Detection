import joblib
from huggingface_hub import hf_hub_download
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import numpy as np

REPO_ID = "akaaan2511/skin_cancer_detection"

# download
svm_path = hf_hub_download(repo_id=REPO_ID, filename="models/multi_svm_model.pkl")
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="models/multi_scaler.pkl")
encoder_path = hf_hub_download(repo_id=REPO_ID, filename="models/multi_label_encoder.pkl")

# load
multi_svm_model = joblib.load(svm_path)
multi_scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# HOG extractor
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


def predict_multi_svm(image):
    features = extract_hog_single(image)
    features_scaled = multi_scaler.transform(features)

    pred_num = multi_svm_model.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    return pred_num, pred_label
