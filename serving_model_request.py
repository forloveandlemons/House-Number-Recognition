
import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image

# from flask_cors import CORS

app = Flask(__name__)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'



@app.route('/house_number_predictor/predict/', methods=['POST'])
def image_classifier():
    # Creating payload for TensorFlow serving request
    payload = {
        "instances": img
    }
    # Making POST request
    res = requests.post('http://localhost:8501/v1/models/house_number_recognition_model:predict', json=json.dump(payload))
    # Decoding results from TensorFlow Serving server
    pred_str = "".join([str(i) for i in pred if i != 10])
    # Returning JSON response to the frontend
    return jsonify()

