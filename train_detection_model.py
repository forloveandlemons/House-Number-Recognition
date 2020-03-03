
X_train = np.load("alldata/detection/data/train.npy")
X_valid = np.load("alldata/detection/data/test.npy")

y_train = np.load("alldata/detection/labels/train.npy")
y_labels = np.load("alldata/detection/labels/test.npy")



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dropout,  Dense
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import pickle

def detection_CNN(X_train, X_valid, y_train, y_valid, modName):
    _, row, col, channel = X_train.shape
    digLen = 5
    numDigits = 11
    epochs = 65
    batch_size = 64

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.keras.backend.set_session(sess)

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    input = tf.keras.Input(shape = (row,col,channel))
    vgg16 = VGG16(include_top = False, weights = 'imagenet')(input)
    vgg16 = Flatten(name = 'flatten')(vgg16)
    vgg16 = Dense(512, activation='relu', name = 'fc_1')(vgg16)
    vgg16 = Dense(64, activation='relu', name = 'fc_2')(vgg16)
    out = Dense(4, activation="linear")(vgg16)
    model = tf.keras.Model(inputs=ptInput, outputs=out)

    model.compile(loss = 'mse',
                  optimizer= "adam",
                  metrics=  ['mae', 'mse'])
    model.summary()
    model_hist = model.fit(x = X_train,
                           y = y_train,
                           batch_size = batch_size,
                           epochs=epochs,
                           verbose=1,
                           shuffle = True,
                           validation_data = (X_valid, y_valid))
    pickle_file = 'models/{}/history.pickle'.format(modName)
    pickle.dump(model_hist.history, open(pickle_file, 'wb'))
    model.save('models/{}/model.h5'.format(modName))



if not os.path.exists('models'):
    os.mkdir("models")

if not os.path.exists("models/detection"):
    os.mkdir("models/detection")

detection_CNN(X_train, X_valid, y_train, y_valid, modName)
