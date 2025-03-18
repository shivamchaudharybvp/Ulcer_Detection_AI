from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load Model
model_path = "C:/Users/shiva/Desktop/finalyear/models/ulcer_model.h5"
model = tf.keras.models.load_model(model_path)
class_labels = ['Mild', 'Moderate', 'Severe']

# Prediction Function
def predict_ulcer(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    
    return class_labels[predicted_class]

# API Route
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    result = predict_ulcer(file_path)
    os.remove(file_path)

    return jsonify({'prediction': result})

# Run Server
if __name__ == '__main__':
    app.run(debug=True)
