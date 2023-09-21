from flask import Flask, render_template, request
import numpy as np
import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), 'models', 'modelo_final.pkl')
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Fiebre = int(request.form['Fiebre'])
    Tos = int(request.form['Tos'])
    Dolor_de_garganta = int(request.form['Dolor_de_garganta'])
    Congestión_nasal = int(request.form['Congestión_nasal'])
    Dificultad_respiratoria = int(request.form['Dificultad_respiratoria'])

    new_samples = np.array([[Fiebre, Tos, Dolor_de_garganta, Congestión_nasal, Dificultad_respiratoria]])
    prediction = model.predict(new_samples)

    mensaje = "La predicción de la enfermedad es: "
    mensaje += prediction[0]

    return render_template('result.html', pred = mensaje)


if __name__ == '__main__':
    app.run()