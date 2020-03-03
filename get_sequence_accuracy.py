import tensorflow as tf
import numpy as np
def get_sequence_accuracy(model_name, input_data, label, data_type):
    model_path = "models/{}/model.h5".format(model_name)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(input_data)
    numsamp = label.shape[0]
    result  = []
    for i in range(5):
        result.append(np.argmax(predictions[i],axis=1).astype('uint8'))
    seqAcc = np.count_nonzero(np.all(np.array(result).T == label[:, 1:6], axis=1)) / np.float(numsamp)
    print("{0:.2%}".format(seqAcc) + " on {} model {} dataset".format(model_name, data_type))


print("loading train tvalidation data")
X_train = np.load("classification_data/train/data.npy")
X_test = np.load("classification_data/test/data.npy")
X_valid = np.load("classification_data/valid/data.npy")
y_train = np.load("classification_data/train/labels.npy")
y_test = np.load("classification_data/test/labels.npy")
y_valid = np.load("classification_data/valid/labels.npy")
for model_name in ["vgg16_scratch", "vgg16_pretrained", "customized"]:
    get_sequence_accuracy(model_name, X_train, y_train, "train")
    get_sequence_accuracy(model_name, X_valid, y_valid, "validation")
    get_sequence_accuracy(model_name, X_test, y_test, "test")

