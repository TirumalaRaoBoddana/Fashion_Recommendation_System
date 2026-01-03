import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from PIL import Image
import warnings
import os
import gdown
warnings.filterwarnings("ignore")

# ------------------ Model ------------------

@st.cache_resource
def load_feature_extractor():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = Model(
        inputs=base_model.input,
        outputs=GlobalAveragePooling2D()(base_model.output)
    )
    return model

@st.cache_data(show_spinner="Downloading model files...")
def download_files_from_gdrive():
    files = {
        "image_embeddings.npy": "1nhCTqucTDy110lC4Q83CwSuPdntxgqao",
        "image_names.npy": "1JS69UFzyrfOZoJZZO8drWf7vRNNtsxmc",
        "mapped_meta_data.csv": "196qOZUTwERTp9c3XRl3avfZn73-AshMh",
        "knn_model.joblib": "1iSC-lVUAqSKp4s_fWSMcfF5NX7MfvHRd",
    }

    for filename, file_id in files.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)


def extract_image_embedding(pil_image, model):
    img = pil_image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    embedding = model.predict(img_array, verbose=0)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding  

@st.cache_data
def load_data():
    try:
        download_files_from_gdrive()
        image_embeddings = np.load("image_embeddings.npy", allow_pickle=True)
        image_names = np.load("image_names.npy", allow_pickle=True).astype(str)
    except Exception as e:
        st.error(f"Loading failed: {e}. Regenerating compatible data...")
        image_embeddings = np.load("image_embeddings.npy", allow_pickle=True)
        image_names = np.load("image_names.npy", allow_pickle=True).astype(str)
    
    meta_data = pd.read_csv("mapped_meta_data.csv")
    meta_data["image_name"] = meta_data["image_name"].astype(str)
    knn = joblib.load("knn_model.joblib")
    return image_embeddings, image_names, meta_data, knn


def predict_info_from_neighbors(top_indices, metadata):
    top_genders = metadata.iloc[top_indices]["gender"]
    predicted_gender = top_genders.mode()[0]  

    # top_master_category=meta_data.iloc[top_indices]["masterCategory"]
    # predicted_master_category=top_master_category.mode()[0]

    top_sub_category=meta_data.iloc[top_indices]["subCategory"]
    predicted_sub_category=top_sub_category.mode()[0]

    top_article_type=meta_data.iloc[top_indices]["articleType"]
    predicted_article_type=top_article_type.mode()[0]
    return (predicted_gender,predicted_sub_category,predicted_article_type)

def recommend_products(
    user_image,
    feature_extractor,
    knn_model,
    image_names,
    metadata,
    top_k=10
):
    # 1️⃣ Extract embedding
    query_embedding = extract_image_embedding(user_image, feature_extractor)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # normalize

    # 2️⃣ Get top 50 neighbors
    distances, indices = knn_model.kneighbors(
        query_embedding,
        n_neighbors=50
    )

    # 3️⃣ Predict gender, subCategory, articleType from top neighbors
    predicted_gender, predicted_subCategory, predicted_articleType = predict_info_from_neighbors(indices[0][:10], metadata)
    st.write(f"Predicted Gender: {predicted_gender} | SubCategory: {predicted_subCategory} | ArticleType: {predicted_articleType}")

    # 4️⃣ Get recommended images from neighbors
    recommended_images = image_names[indices[0]]
    recommendations = metadata[metadata["image_name"].isin(recommended_images)]

    # 5️⃣ Filter recommendations based on predicted info
    filtered_recommendations = recommendations[
        (recommendations["gender"] == predicted_gender) &
        (recommendations["articleType"] == predicted_articleType) &
        (recommendations["subCategory"] == predicted_subCategory)
    ]

    return filtered_recommendations.head(top_k)

def display_cards(recommendations, cards_per_row=5):
    if recommendations.shape[0] == 0:
        st.warning("No recommendations found.")
        return

    # Split recommendations into chunks of size cards_per_row
    chunks = [recommendations.iloc[i:i+cards_per_row] for i in range(0, recommendations.shape[0], cards_per_row)]

    for chunk in chunks:
        cols = st.columns(len(chunk))  # create columns for this row
        for idx, row in enumerate(chunk.itertuples()):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #e0e0e0;
                        border-radius:10px;
                        padding:10px;
                        text-align:center;
                        box-shadow: 2px 2px 12px #f0f0f0;
                        margin-bottom:10px;
                    ">
                        <img src="{row.link}" width="180" style="border-radius:10px"/>
                        <p style="margin:5px 0 0 0; font-weight:bold;">{row.productName if 'productName' in row._fields else ''}</p>
                        <p style="margin:2px 0 0 0; color:#555">{row.masterCategory} | {row.subCategory}</p>
                        <p style="margin:2px 0 0 0; color:#999">{row.gender if 'gender' in row._fields else ''}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )



if __name__ == "__main__":
    st.set_page_config(page_title="Fashion Recommendation System", layout="wide")
    st.title("👗 Fashion Recommendation System")

    feature_extractor = load_feature_extractor()
    image_embeddings, image_names, meta_data, knn = load_data()

    uploaded_file = st.file_uploader(
        "Upload a fashion product image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        user_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(user_image, width=250)

        with st.spinner("Finding similar products..."):
            recommendations = recommend_products(
                user_image=user_image,
                feature_extractor=feature_extractor,
                knn_model=knn,
                image_names=image_names,
                metadata=meta_data
            )

        st.subheader("Recommended Products")

        display_cards(recommendations, cards_per_row=5)



