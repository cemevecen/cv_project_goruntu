import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Page Config
st.set_page_config(
    page_title="AI Vision Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-card {
        padding: 24px;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2e7d32;
        margin-top: 20px;
    }
    h1 {
        color: #1b5e20;
        font-family: 'Inter', sans-serif;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load Model with Caching
@st.cache_resource
def load_vision_model():
    model_path = "models/model.h5"
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

model = load_vision_model()

# Header
st.title("🖼️ AI Vision Classifier")
st.markdown("<p style='text-align: center; font-size: 1.2em;'>Gelişmiş Yapay Zeka ile Görüntü Sınıflandırma</p>", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader("Bir Resim Seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Yüklenen Görüntü', use_container_width=True)
    
    with col2:
        if model is not None:
            with st.spinner('Analiz yapılıyor...'):
                # Preprocess image for MobileNetV2
                processed_image = image.resize((224, 224))
                img_array = np.array(processed_image)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prediction
                predictions = model.predict(img_array)
                decoded = decode_predictions(predictions, top=3)[0]

                # Results UI
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.subheader(f"✅ Tahmin: {decoded[0][1].replace('_', ' ').capitalize()}")
                
                st.write("---")
                st.markdown("**Olasılık Detayları:**")
                for _, label, prob in decoded:
                    label_clean = label.replace('_', ' ').capitalize()
                    st.write(f"{label_clean}")
                    st.progress(float(prob))
                    st.caption(f"Doğruluk: %{prob*100:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Model yüklenemediği için analiz yapılamıyor.")

else:
    # Welcome / Info state
    st.info("👆 Lütfen analiz etmek istediğiniz bir resim dosyasını yukarıdan sürükleyip bırakın veya seçin.")
    
    # Showcase cards or instructions
    st.markdown("""
    ### Nasıl Çalışır?
    1. Bir fotoğraf yükleyin.
    2. Arka planda MobileNetV2 tabanlı modelimiz görüntüyü işler.
    3. En olası 3 kategoriyi ve güven skorlarını anında görün.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>© 2026 Nejdet TUT - Tüm Hakları Saklıdır.</p>", unsafe_allow_html=True)
