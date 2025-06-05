import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plant Disease Detection")

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('plant_disease_detection.keras')
        return model
    except:
        try:
            model = keras.models.load_model('plant_disease_detection.keras')
            return model
        except:
            st.error("Model file not found! Please train the model first.")
            return None

def preprocess_image(image):
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized)
    img_normalized = img_array.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, img_resized

def check_if_plant_like(image):
    img_small = image.resize((32, 32))
    img_array = np.array(img_small)
    
    green_pixels = np.sum(img_array[:,:,1] > img_array[:,:,0]) + np.sum(img_array[:,:,1] > img_array[:,:,2])
    total_pixels = img_array.shape[0] * img_array.shape[1] * 2
    
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.3

def main():
    st.title("🌱 Plant Disease Detection")
    st.write("Upload a plant leaf image to check if it's healthy or diseased")
    
    model = load_model()
    if model is None:
        st.stop()
    
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        if not check_if_plant_like(image):
            st.warning("This might not be a plant image. Results may not be accurate.")
        
        if st.button("Analyze Plant"):
            with st.spinner("Analyzing..."):
                processed_image, resized_image = preprocess_image(image)
                
                with col2:
                    st.image(resized_image, caption="Processed Image (64x64)", use_container_width=True)
                
                prediction_prob = model.predict(processed_image, verbose=0)[0][0]
                predicted_class = 1 if prediction_prob > 0.5 else 0
                confidence = max(prediction_prob, 1 - prediction_prob)
                
                st.markdown("---")
                
                if predicted_class == 0:
                    st.success(f"HEALTHY PLANT")
                    st.write(f"Confidence: {confidence:.1%}")
                    if confidence > 0.8:
                        st.write("High confidence result")
                    elif confidence > 0.6:
                        st.write("Medium confidence result")
                    else:
                        st.write("Low confidence result")
                else:
                    st.error(f"DISEASED PLANT")
                    st.write(f"Confidence: {confidence:.1%}")
                    if confidence > 0.8:
                        st.write("High confidence result")
                    elif confidence > 0.6:
                        st.write("Medium confidence result")
                    else:
                        st.write("Low confidence result")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                categories = ['Healthy', 'Diseased']
                probabilities = [1 - prediction_prob, prediction_prob]
                colors = ['green', 'red']
                
                bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Results')
                
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
                with st.expander("Tips for better results"):
                    st.write("""
                    - Use clear, well-lit photos of plant leaves
                    - Avoid blurry or very dark images
                    """)

if __name__ == "__main__":
    main()