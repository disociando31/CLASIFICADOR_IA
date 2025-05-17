from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
model_path = 'modelo_clasificador_ia.h5'  # Asegúrate de tenerlo cuando entrenes
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model = load_model(model_path)  # Descomenta esto cuando tengas el modelo

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No se proporcionó imagen')

    img_file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    # Procesar imagen (cuando tengas el modelo)
    # img = load_img(img_path, target_size=(224, 224))  # Ajusta a tu modelo
    # img_array = img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # prediction = model.predict(img_array)
    # label = np.argmax(prediction)

    # Simulación temporal
    label = "Simulación: clase_1"

    return render_template('index.html', prediction=f'Resultado: {label}', image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
