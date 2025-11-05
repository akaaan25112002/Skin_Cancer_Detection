import streamlit as st
from PIL import Image
from src.svm_inference import predict_svm
from src.efficientnet_inference import predict_efficientnet
from src.multi_svm_inference import predict_multi_svm
from src.efficientnet_multi_inference import predict_efficientnet_multi

# ==========================================
# LABEL MAP
# ==========================================
name_map = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "vasc":  "Vascular Lesion"
}

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown(
    """
    <style>
    .result-card {
        background-color: #111827;
        color: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #333;
        margin-top: 10px;
    }
    .benign {
        background-color: #16a34a !important;
    }
    .malignant {
        background-color: #dc2626 !important;
    }
    .prob-text {
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# MAIN
# ==========================================
def main():

    st.title("🔬 Skin Cancer Detection")
    st.write("Analyze skin lesion images using Machine Learning & Deep Learning.")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    task = st.sidebar.selectbox(
        "Task Mode",
        ["Binary Classification", "Multi-class Classification"]
    )
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["SVM", "EfficientNet B2"]
    )

    st.divider()

    # Upload
    uploaded = st.file_uploader(
        "📤 Upload a skin lesion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("🔎 Prediction Result")

            # ==================================
            # BINARY
            # ==================================
            if task == "Binary Classification":

                if model_type == "SVM":
                    pred, prob = predict_svm(img)
                else:
                    pred, prob = predict_efficientnet(img)

                label = "Malignant" if pred == 1 else "Benign"
                cls_color = "malignant" if pred == 1 else "benign"

                # NICE RESULT CARD
                st.markdown(
                    f"""
                    <div class="result-card {cls_color}">
                        <h3 style="margin:0;">{label}</h3>
                        <p class="prob-text">Probability (Malignant): {prob:.3f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ==================================
            # MULTI
            # ==================================
            elif task == "Multi-class Classification":

                # -------------------- SVM --------------------
                if model_type == "SVM":
                    num, pred_label = predict_multi_svm(img)
                    friendly = name_map[pred_label]

                    st.success(f"Prediction: {friendly} ({pred_label})")

                # ---------------- EfficientNet ----------------
                elif model_type == "EfficientNet B2":
                    result = predict_efficientnet_multi(img)

                    friendly = name_map[result["pred_class"]]

                    st.success(
                        f"Prediction: **{friendly}** ({result['pred_class']})"
                    )
                    st.write(f"Confidence: {result['confidence']:.3f}")

                    st.write("Top-3 predictions:")
                    for cls, prob in result["topk"]:
                        st.write(f"- {name_map[cls]} → **{prob:.3f}**")

        st.divider()

    else:
        st.info("👆 Upload a skin image to get started.")

    # Footer
    st.markdown(
        """
        <br><hr>
        <p style='text-align:center; color:gray;'>
            Developed for Skin Cancer Research · EfficientNet-B2 & SVM
        </p>
        """,
        unsafe_allow_html=True,
    )


# EXEC
if __name__ == "__main__":
    main()