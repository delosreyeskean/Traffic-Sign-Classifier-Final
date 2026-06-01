import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the model
model = tf.keras.models.load_model('traffic_classifier.keras')

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

# 2. Prediction Function
def classify_traffic_sign(img):
    # Preprocess the image
    img = img.resize((30, 30))
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
        
    # Expand dimensions for the batch format (1, 30, 30, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)[0]
    
    # Format results for Gradio (returns a dictionary of classes and their confidences)
    confidences = {classes[i]: float(predictions[i]) for i in range(43)}
    return confidences

# 3. Create the Web UI
interface = gr.Interface(
    fn=classify_traffic_sign,
    inputs=gr.Image(type="pil", label="Upload Traffic Sign"),
    outputs=gr.Label(num_top_classes=3, label="CNN Predictions"),
    title="🚦 Traffic Sign Recognition Web App",
    description="Upload a picture of a German traffic sign to see the Convolutional Neural Network's prediction.",
    theme="default"
)

# 4. Launch with a public live link
if __name__ == "__main__":
    interface.launch(share=True)