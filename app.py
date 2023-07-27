import pickle5 as pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json
import os

preprocessed_data_path = os.path.join("pickle_files","preprocessor.pkl")
with open(preprocessed_data_path,"rb") as f:
    transform_data = pickle.load(f)

with open('knn.pkl','rb') as f1:
    knnmodel= pickle.load(f1)

app = Flask(__name__)

@app.route('/predictdata',method = ['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data = request.form
        df_data = pd.read_json(data)
        transfomred_input_data = transform_data.transform(df_data)
        results = knnmodel.predict(transfomred_input_data)
    return render_template('home.html',results = results[0])


if __name__=="__main__":
    app.run(debug=True)