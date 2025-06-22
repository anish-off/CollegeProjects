from flask import Flask, request, render_template
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define paths
MODEL_PATH = 'best_model.h5'  # Adjust if .h5 file is elsewhere
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# Image preprocessing function
def preprocess_image(image_path):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if file:
        # Save the uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Preprocess and predict
        img_array = preprocess_image(filename)
        prediction = model.predict(img_array)
        probability = prediction[0][0]
        class_label = 'Non-Autistic' if probability > 0.5 else 'Autistic'
        confidence = probability if probability > 0.5 else 1 - probability
        
        # Render result
        return render_template('result.html', 
                             class_label=class_label, 
                             confidence=confidence*100, 
                             image_path=filename)

if __name__ == '__main__':
    app.run(debug=True)