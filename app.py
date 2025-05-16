from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
model = load_model('modelo_clasificador_ia.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    path = os.path.join('static/uploads', img_file.filename)
    img_file.save(path)

    img = load_img(path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = 'IA' if prediction > 0.5 else 'Real'

    return jsonify({'prediction': label, 'confidence': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)