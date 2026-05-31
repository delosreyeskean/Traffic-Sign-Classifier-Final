import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained neural network model
MODEL_PATH = 'traffic_classifier.h5'
model = load_model(MODEL_PATH)

# Map the 43 numerical classes to human-readable names
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road works',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

@app.route('/', methods=['GET'])
def index():
    # Render the main frontend upload page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was actually uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open image file using PIL
        img = Image.open(file.stream)
        
        # PREPROCESSING: Convert to RGB, resize to 30x30, convert to numpy array
        img = img.convert('RGB')
        img = img.resize((30, 30))
        img_array = np.array(img)
        
        # PREPROCESSING: Normalize pixel values to be between 0 and 1
        img_array = img_array / 255.0
        
        # Reshape array to batch format (1, 30, 30, 3) so model can process it
        img_array = np.expand_dims(img_array, axis=0)
        
        # Execute the prediction
        predictions = model.predict(img_array)
        predicted_class_id = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        
        # Get human-readable label
        predicted_label = CLASSES.get(predicted_class_id, "Unknown Sign")
        
        return jsonify({
            'class_id': predicted_class_id,
            'prediction': predicted_label,
            'confidence': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask server
    app.run(debug=True, port=5000)