import streamlit as st
from PIL import Image
from src.svm_inference import predict_svm
from src.efficientnet_inference import predict_efficientnet
from src.multi_svm_inference import predict_multi_svm

name_map = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "vasc":  "Vascular Lesion"
}
# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# MAIN
# ==========================
def main():

    st.title("🔬 Skin Cancer Detection")
    st.markdown(
        """
        This application allows image-based skin cancer prediction using:
        - ✅ **SVM (HOG-based Machine Learning)**
        - ✅ **EfficientNet-B2 (Deep Learning)**

        You can switch between **Binary** and **Multi-class** classification modes.
        """
    )

    # Sidebar selector
    st.sidebar.header("⚙️ Settings")

    task = st.sidebar.selectbox(
        "Task Type",
        ["Binary Classification", "Multi-class Classification"]
    )

    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["SVM", "EfficientNet B2"]
    )

    st.divider()

    # Upload section
    uploaded = st.file_uploader(
        "📤 Upload skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        
        # Display image
        st.image(img, caption="Uploaded Image", width=350)

        # Predict
        st.subheader("🔎 Prediction Result")

        if task == "Binary Classification":

            if model_type == "SVM":
                pred, prob = predict_svm(img)
            else:
                pred, prob = predict_efficientnet(img)

            label = "Malignant (Cancer)" if pred == 1 else "Benign"
            st.write(f"**Prediction:** {label}")
            st.write(f"**Probability (Malignant):** {prob:.4f}")

        elif task == "Multi-class Classification":
            
            if model_type == "SVM":
                num, pred_label = predict_multi_svm(img)
            
            friendly = name_map[pred_label]
            st.write(f"Prediction Code: {pred_label}")
            st.write(f"Diagnosis: {friendly}")
            
        else:
            st.warning("🚧 Multi-class mode under development.")

    else:
        st.info("👆 Upload a skin image to begin.")


# EXEC
if __name__ == "__main__":
    main()
