import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")

# 1. Load the model (Cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    # Load the H5 model file
    model = tf.keras.models.load_model('traffic_classifier.h5')
    return model

model = load_model()

# Dictionary containing the 43 standard GTSRB classes
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 tons',
    11: 'Right-of-way at intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

st.title("🚦 Traffic Sign Recognition App")
st.write("Upload a picture of a traffic sign, and the Convolutional Neural Network (CNN) will classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)

    st.write("Analyzing...")

    # 2. Data Preprocessing
    # Resize to match the input shape of the model
    image = image.resize((30, 30)) 
    image_array = np.array(image)
    
    # Ensure image has 3 channels (RGB). Drop alpha channel if PNG
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
        
    # Expand dimensions to match model's expected batch format (1, 30, 30, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Normalize if your model was trained on normalized data (uncomment if needed)
    # image_array = image_array / 255.0 

    # 3. Model Inference (Prediction)
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    # 4. Output
    st.success(f"**Prediction:** {classes.get(class_index, 'Unknown')}")
    st.info(f"**Confidence:** {confidence:.2%}")