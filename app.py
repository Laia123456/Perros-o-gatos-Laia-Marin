import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Carrega el model
model = load_model("model_gats_gossos.h5")

# UI
st.title("Classificador de Gats i Gossos 🐱🐶")
st.markdown("Puja una imatge i el model predirà si és un **gat** o un **gos**.")

uploaded_file = st.file_uploader("Puja una imatge", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Imatge carregada', use_column_width=True)

    # Preprocessament
    img_resized = img.resize((100, 100))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # Resultat
    if prediction > 0.5:
        st.success(f"És un **GOS** 🐶 ({prediction:.2f})")
    else:
        st.success(f"És un **GAT** 🐱 ({1 - prediction:.2f})")
