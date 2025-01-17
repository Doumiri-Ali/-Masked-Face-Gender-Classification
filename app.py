import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image


IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CLASSES = 2  # Male and Female


st.set_page_config(page_title="Gender Classification App", layout="wide")


@st.cache_resource
def load_model():
    try:
        try:
            model = tf.keras.models.load_model("gender_classification_model.keras")
            st.success("Full model loaded successfully!")
            return model
        except Exception as e:
            st.warning(f"Could not load full model: {str(e)}. Trying to load weights...")
            
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])
            
            model = tf.keras.models.Sequential([
                data_augmentation,
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
            
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                
                tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(NUM_CLASSES)  # 2 classes: Male and Female
            ])

            model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            
            try:
                model.load_weights("gender_classification_weights.h5")
                st.success("Model weights loaded successfully!")
            except Exception as e:
                st.error(f"Could not load weights: {str(e)}. Using untrained model.")
            
            return model
            
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.error("Failed to create the model. Please check the error message above.")
    st.stop()

def preprocess_image(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3 and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_gender(image_array):
    predictions = model.predict(image_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    predicted_class = np.argmax(score)
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence

st.title("Gender Classification App")

tab1, tab2 = st.tabs(["Upload Image", "Webcam"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Classify Gender"):
            with st.spinner("Processing..."):
                processed_image = preprocess_image(image)
                predicted_class, confidence = predict_gender(processed_image)
                
                st.write(f"Predicted Gender: {'Female' if predicted_class == 0 else 'Male'}")
                st.write(f"Confidence: {confidence:.2f}%")

with tab2:
    st.header("Webcam Input")
    
    camera = st.camera_input("Take a photo")
    
    if camera is not None:
        image = Image.open(camera)
        image = np.array(image)
        st.image(image, use_container_width=True)

        with st.spinner("Processing..."):
            processed_image = preprocess_image(image)
            predicted_class, confidence = predict_gender(processed_image)
            
            st.write(f"Predicted Gender: {'Female' if predicted_class == 0 else 'Male'}")
            st.write(f"Confidence: {confidence:.2f}%")