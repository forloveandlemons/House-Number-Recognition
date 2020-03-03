

print("loading train tvalidation data")
X_train = np.load("classification_data/train/data.npy")
X_test = np.load("classification_data/test/data.npy")
X_valid = np.load("classification_data/valid/data.npy")

y_train = np.load("classification_data/train/labels.npy")
y_test = np.load("classification_data/test/labels.npy")
y_valid = np.load("classification_data/valid/labels.npy")

reshaped_y_train = [(np.reshape(y_train[:len(y_train), i], (len(y_train), 1))).astype('uint8') for i in range(1, 6)]
reshaped_y_valid = [(np.reshape(y_valid[:len(y_valid), i], (len(y_valid), 1))).astype('uint8') for i in range(1, 6)]
reshaped_y_test = [(np.reshape(y_test[:len(y_test), i], (len(y_test), 1))).astype('uint8') for i in range(1, 6)]


# ## 2. Classification Training

# ## 2.1 Self designed model

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization

def customized(X_train, X_valid, reshaped_y_train, reshaped_y_valid, model_name):
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    _,row, col,channel = X_train.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 50
    batch_size = 64

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.keras.backend.set_session(sess)

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    _,row, col,channel = X_train.shape

    input = tf.keras.Input(shape=(row,col,channel), name='customModel')
    conv1 = Conv2D(16, (3, 3),activation='relu',padding='same', name = 'conv1_a')(input)
    conv1 = Conv2D(16, (3, 3), activation ='relu', padding='same',name = 'conv1_b')(conv1)
    conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_a')(conv1)
    conv2 = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_b')(conv2)
    conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Dropout(0.5)(conv2)

    conv3 = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv3_a')(conv2)
    conv3 = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv3_b')(conv3)
    conv3 = BatchNormalization(axis=-1)(conv3)
    conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv4_a')(conv3)
    conv4 = Conv2D(64, (3, 3), activation ='relu', padding='same', name = 'conv4_b')(conv4)
    conv4 = BatchNormalization(axis=-1)(conv4)
    conv4 = MaxPooling2D((2, 2), strides= 1)(conv4)

    conv5 = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv5_a')(conv4)
    conv5 = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv5_b')(conv5)
    conv5 = BatchNormalization(axis=-1)(conv5)
    conv5 = MaxPooling2D(pool_size=(2, 2),strides = 1)(conv5)
    conv5 = Dropout(0.5)(conv5)

    fc = Flatten()(conv5)
    fc = Dense(64, activation='relu', name = 'fc1')(fc)
    fc = Dense(64, activation='relu', name = 'fc2')(fc)

    d1 = Dense(11, activation='softmax')(fc)
    d2 = Dense(11, activation='softmax')(fc)
    d3 = Dense(11, activation='softmax')(fc)
    d4 = Dense(11, activation='softmax')(fc)
    d5 = Dense(11, activation='softmax')(fc)
    out = [d1, d2, d3, d4, d5]

    model = tf.keras.Model(inputs = input, outputs=out)
    model.compile(loss = 'sparse_categorical_crossentropy',
                      optimizer= "adam",
                      metrics=  ['accuracy'])
    model.summary()
    callback = []
    es = tf.keras.callbacks.EarlyStopping(monitor= 'loss',  #'dig1_loss',
                                       min_delta=0.00001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    model_hist = model.fit(x = X_train,
                            y = reshaped_y_train,
                            batch_size = batch_size,
                            epochs = 1,
                            verbose=1,
                            shuffle = True,
                            validation_data = (X_valid, reshaped_y_valid),
                            callbacks= callback)

    pickle_file = 'models/{}/history.pickle'.format(model_name)
    pickle.dump(model_hist.history, open(pickle_file, 'wb'))
    model.save("models/{}/model.h5".format(model_name))


# ## 2.2 Train VGG16 from scratch

import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16

def vgg16_scratch(X_train, X_valid, reshaped_y_train, reshaped_y_valid, model_name):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.keras.backend.set_session(sess)
    
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    _,row, col,channel = X_train.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 50
    batch_size = 64
    
    input = tf.keras.Input(shape = (row,col,channel))
    vgg16 = VGG16(include_top = False, weights = None)(input)
    vgg16 = Flatten()(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    vgg16 = Dense(64, activation='relu')(vgg16)
    vgg16 = Dropout(0.5)(vgg16)
    
    d1 = Dense(11, activation='softmax')(vgg16)
    d2 = Dense(11, activation='softmax')(vgg16)
    d3 = Dense(11, activation='softmax')(vgg16)
    d4 = Dense(11, activation='softmax')(vgg16)
    d5 = Dense(11, activation='softmax')(vgg16)
    out = [d1, d2, d3, d4, d5]
    
    model = tf.keras.Model(inputs=input, outputs = out)

    callback = []
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    es = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss',
                                       min_delta=0.00000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    model.summary()
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer= optim,
                  metrics=['accuracy'])

    model_hist = model.fit(x = X_train,
                             y = reshaped_y_train,
                             batch_size = batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle = True,
                             validation_data = (X_valid, reshaped_y_valid),
                             callbacks = callback)
    pickle_file = 'models/{}/history.pickle'.format(model_name)
    pickle.dump(model_hist.history, open(pickle_file, 'wb'))
    model.save("models/{}/model.h5".format(model_name))


# ## 2.3 Pretrained VGG 16

def vgg16_pretrained(X_train, X_valid, reshaped_y_train, reshaped_y_valid, model_name):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.keras.backend.set_session(sess)
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    _,row, col,channel = X_train.shape
    digLen = 5
    numDigits = 11
    epochs = 50
    batch_size = 64
    input = tf.keras.Input(shape = (row,col,channel))
    vgg16 = VGG16(include_top = False, weights='imagenet')(input)
    vgg16 = Flatten(name = 'flatten')(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    vgg16 = Dense(64, activation='relu')(vgg16)
    d1 = Dense(11, activation='softmax')(vgg16)
    d2 = Dense(11, activation='softmax')(vgg16)
    d3 = Dense(11, activation='softmax')(vgg16)
    d4 = Dense(11, activation='softmax')(vgg16)
    d5 = Dense(11, activation='softmax')(vgg16)
    out = [d1, d2, d3, d4, d5]
    model = tf.keras.Model(inputs=input, outputs=out)
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer= optim,
                  metrics=  ['accuracy'])
    model.summary()
    callback = []
    es = tf.keras.callbacks.EarlyStopping(monitor= 'loss',
                                       min_delta=0.000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    model_hist = model.fit(x = X_train,
                         y = reshaped_y_train,
                         batch_size = batch_size,
                         epochs=epochs,
                         verbose=1,
                         shuffle = True,
                         validation_data = (X_valid, reshaped_y_valid),
                         callbacks= callback)
    pickle_file = 'models/{}/history.pickle'.format(model_name)
    pickle.dump(model_hist.history, open(pickle_file, 'wb'))
    model.save("models/{}/model.h5".format(model_name))


import os
if not os.path.exists("models"):
    os.mkdir("models")

for model_name in ["customized", "vgg16_pretrained", "vgg16_scratch"]:
    if not os.path.exists("models/{}".format(model_name)):
        os.mkdir("models/{}".format(model_name))


customized(X_train, X_valid, reshaped_y_train, reshaped_y_valid, "customized")
vgg16_scratch(X_train, X_valid, reshaped_y_train, reshaped_y_valid, "vgg16_scratch")
vgg16_pretrained(X_train, X_valid, reshaped_y_train, reshaped_y_valid, "vgg16_pretrained")

