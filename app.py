import json
import numpy as np
import requests
import cv2
from flask import Flask, jsonify, request, redirect, render_template

app = Flask(__name__, template_folder='public')
detection_input_size = 128
classification_input_size = 32

app.config['IMAGE_UPLOADS'] = 'image_uploads'
HOUSE_NUMBER_PREDICT_API_ENDPOINT = "http://localhost:5000/house_number_predict/predict/"


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files['image']
            image_data = image.read()
            data = {'image_data': image_data}
            # sending post request and saving response as response object
            res = requests.post(url=HOUSE_NUMBER_PREDICT_API_ENDPOINT, files=data)
            return res.json()
            # return redirect(request.url)
    return render_template('upload_image.html')


@app.route('/house_number_predict/predict/', methods=['POST'])
def house_number_predictor():
    file = request.files['image_data']
    img_str = file.stream.read()
    np_arr = np.fromstring(img_str, np.uint8)
    img_raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    copy_raw = np.copy(img_raw)
    img = cv2.resize(img_raw, (detection_input_size, detection_input_size), interpolation=cv2.INTER_AREA)
    new_img_list = np.expand_dims(img, axis=0).tolist()
    detection_payload = {"instances": new_img_list}

    detection_res = requests.post('http://localhost:8501/v1/models/house_number_recognition_detection_model:predict',
                                  data=json.dumps(detection_payload))
    (x, y, w, h) = detection_res.json()['predictions'][0]
    # add some padding
    y = int(y * 0.7)
    x = int(x * 0.7)
    h = int(h * 1.35)
    w = int(w * 1.35)
    raw_x = int(img_raw.shape[1] / img.shape[1] * x)
    raw_y = int(img_raw.shape[0] / img.shape[0] * y)
    raw_w = min(int(img_raw.shape[1] / img.shape[1] * w), int(img_raw.shape[1] - raw_x - 1))
    raw_h = min(int(img_raw.shape[0] / img.shape[0] * h), int(img_raw.shape[0] - raw_y - 1))
    patch = copy_raw[raw_y:raw_y + raw_h, raw_x:raw_x + raw_w]
    patch = cv2.resize(patch, (classification_input_size, classification_input_size))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.reshape((1, classification_input_size, classification_input_size, 3))
    classification_payload = {"instances": patch.tolist()}
    classification_res = requests.post('http://localhost:8501/v1/models/house_number_recognition_model:predict',
                                       data=json.dumps(classification_payload))
    raw_res = classification_res.json()['predictions'][0]


    res = []
    for key in [""] + ["_" + str(i) for i in range(1, 5)]:
        new_key = "dense" + key + "/Softmax:0"
        predicted_digit = np.argmax(raw_res[new_key])
        if predicted_digit != 10:
            res.append(str(np.argmax(raw_res[new_key])))
    text = "".join(res)
    return_val = {"predicted_house_number":text}
    return jsonify(return_val)