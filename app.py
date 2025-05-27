from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Configuración de Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo entrenado
model = load_model('modelo_clasificador_ia.h5')

# Verificar tipo de archivo permitido
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función de predicción
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
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

# ===== Rutas Flask =====

# Página principal (Resumen del proyecto y metodología)
@app.route('/')
def index():
    return render_template('index.html')

# Página de objetivos
@app.route('/objetivos')
def objetivos():
    return render_template('objetivos.html')

# Página de planteamiento del problema y justificación
@app.route('/problema')
def problema():
    return render_template('problema.html')

# Página del modelo (carga y predicción de imagen)
@app.route('/modelo', methods=['GET', 'POST'])
def modelo():
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

    return render_template('modelo.html', result=result, filename=filename, value=prediction_value)

# Ejecutar la app
if __name__ == '__main__':
    app.run(debug=True)
