import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pickle


detection_model = load_model('models/detection/model.h5')
classification_model = load_model("models/vgg16_scratch/model.h5")


detection_input_size = 128
classification_input_size = 32
for i in range(1, 6):
    path = "graded_input/{}.png".format(str(i))
    img_raw = cv2.imread(path)
    copy_raw = np.copy(img_raw)
    img     = cv2.resize(img_raw, (detection_input_size, detection_input_size), interpolation=cv2.INTER_AREA)
    (x, y, w, h) = detection_model.predict(np.expand_dims(img, axis=0))[0]
    # add some padding
    y = int(y * 0.7)
    x = int(x * 0.7)
    h = int(h * 1.35)
    w = int(w * 1.35)
    # find boxes in the original image
    raw_x = int(img_raw.shape[1]/img.shape[1] * x)
    raw_y = int(img_raw.shape[0]/img.shape[0] * y)
    raw_w = min(int(img_raw.shape[1]/img.shape[1] * w), int(img_raw.shape[1] - raw_x - 1))
    raw_h = min(int(img_raw.shape[0]/img.shape[0] * h), int(img_raw.shape[0] - raw_y - 1))
    # draw rectangle
    cv2.rectangle(img_raw, (raw_x, raw_y), (raw_x+raw_w, raw_y+raw_h), (0, 255, 0), 1)
    patch = copy_raw[raw_y:raw_y+raw_h, raw_x:raw_x+raw_w]
    patch = cv2.resize(patch, (classification_input_size, classification_input_size))
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.reshape((1, classification_input_size, classification_input_size, 3))
    # detection
    number = classification_model.predict(patch)
    value = []
    num_digit = 0
    for n in number:
        temp = np.argmax(n, axis=-1)
        value.append(temp[0])
        if temp != 10:
            num_digit += 1
    sequence = [str(each) for each in [num_digit] + value]
    text = "[{}]".format(",".join(sequence))
    cv2.putText(img_raw, text,(5, 20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255), lineType=1)
    print("graded_input/{}.png predicted digits".format(str(i)))
    print(sequence)
    cv2.imwrite("graded_images/{}".format(path.split("/")[-1]), img_raw)
