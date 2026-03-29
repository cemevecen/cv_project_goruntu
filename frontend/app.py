#full kod
import streamlit as st
import requests
from PIL import Image
import io

# Page Config
st.set_page_config(
    page_title="AI Vision Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTitle {
        color: #2e7d32;
        text-align: center;
        font-family: 'Outfit', sans-serif;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🖼️ AI Vision Classifier")
st.markdown("### Bir görüntü yükleyin, gelişmiş yapay zeka modelimiz ne olduğunu tahmin etsin.")

uploaded_file = st.file_uploader("Bir Resim Seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)

    # Progress bar for better UX
    with st.spinner('Tahmin ediliyor...'):
        try:
            # Prepare file for requests
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            files = {"file": ("image.jpg", byte_im, "image/jpeg")}

            # Send to Backend
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 200:
                data = response.json()
                prediction = data["prediction"]
                details = data.get("details", [])

                st.success(f"**Tahmin:** {prediction.capitalize()}")

                # Show top 3 results
                st.write("### Detaylar:")
                for res in details:
                    st.write(f"- **{res['label'].capitalize()}**: {res['probability']:.2%}")
                    st.progress(res['probability'])
            else:
                st.error("Sunucu tarafında bir hata oluştu. Backend'in çalıştığından emin olun.")
        except Exception as e:
            st.error(f"Hata: {e}")
else:
    st.info("Lütfen bir resim dosyası yükleyin.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Nejdet TUT - CV Project - 2026</p>", unsafe_allow_html=True)

