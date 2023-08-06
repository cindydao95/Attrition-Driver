import pickle
from flask import Flask, request, app, render_template

import pandas as pd
import os

preprocessed_data_path = "preprocessor.pkl"
with open(preprocessed_data_path,"rb") as f:
    transform_data = pickle.load(f)

model_path  = "knn.pkl"
with open(model_path,'rb') as f1:
    knnmodel= pickle.load(f1)

app = Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')


@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('index.html')
    else:
        data = request.form
        input_data_df = pd.DataFrame(data,index=[0])
        transfomred_input_data = transform_data.transform(input_data_df)
        results = knnmodel.predict(transfomred_input_data)
    return render_template('index.html',results = results[0])


if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)