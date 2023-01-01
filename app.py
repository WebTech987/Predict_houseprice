import json
import pickle5 as pickle
#from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
#import StandardScaler as scalar

app = Flask(__name__)
## load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    scalerfile = 'scaling.pkl'
    scalar = pickle.load(open(scalerfile, 'rb'))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    scalerfile = 'scaling.pkl'
    scalar = pickle.load(open(scalerfile, 'rb'))
    final_input=scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="This is the predicted houseprice value {}".format(output))

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
