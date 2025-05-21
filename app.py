# === app.py actualizado (compatible con softmax y 2 clases) ===
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo softmax
model = load_model('modelo_clasificador_ia.h5')

# Verificar extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predecir imagen
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Predicción: clase {predicted_class}, confianza: {confidence}")

    if predicted_class == 1:
        return "Imagen Generada por IA", confidence
    else:
        return "Imagen Real", confidence

# Ruta principal
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None
    prediction_value = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, prediction_value = predict_image(filepath)
            filename = file.filename
        else:
            result = "Archivo no válido. Solo se permiten PNG, JPG o JPEG."

    return render_template('index.html', result=result, filename=filename, value=prediction_value)

@app.route('/objetivos')
def objetivos():
    return render_template('objetivos.html')

@app.route('/problema')
def problema():
    return render_template('problema.html')

@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

if __name__ == '__main__':
    app.run(debug=True)
