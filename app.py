import pickle
from flask import Flask , request , jsonify , render_template
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## ridge model pkl
ridge_model = pickle.load(open('Models//ridge_model.pkl' , 'rb'))
Standard_scaler = pickle.load(open('Models//scaler_model.pkl' , 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData' , methods=['POST' , 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        temp = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))
        
        new_data_scaled = Standard_scaler.transform([[temp , rh , ws , rain , ffmc , dmc, isi, classes , region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html' , results = result[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0" , port=5000 , debug=True)
