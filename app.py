# app.py
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo
model = load_model('model/cnn_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            result = "Imagen Generada por IA" if prediction >= 0.5 else "Imagen Real"
            filename = file.filename
    return render_template('index.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
