import pickle
import os
import matplotlib.pyplot as plt


def show_training_history(model_name, metric="accuracy"):
    with open("models/{}/history.pickle".format(model_name), "rb") as handle:
        history = pickle.load(handle)
    digit_layers = {"customized" : ["dense", "dense_1", "dense_2", "dense_3", "dense_4"],
                    "vgg16_pretrained": ["dense_5", "dense_6", "dense_7", "dense_8", "dense_9"],
                    "vgg16_scratch": ["dense_93", "dense_94", "dense_95", "dense_96", "dense_97"]}
    plt.figure()
    if metric == "accuracy":
        plt.ylim([0.5, 1])
        for layer in digit_layers[model_name]:
            plt.plot(history['val_{}_acc'.format(layer)])
        plt.title('{} model validation accuracy'.format(model_name))
        plt.ylabel('accuracy')
    elif metric == "loss":
        plt.ylim([0, 1])
        for layer in digit_layers[model_name]:
            plt.plot(history['{}_loss'.format(layer)])
        plt.title('{} model training loss'.format(model_name))
        plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['digit1', 'digit2', 'digit3', 'digit4', 'digit5'], loc='lower right')
    plt.savefig('plots/{}_{}.png'.format(model_name, metric))
    # plt.show()
    plt.close()

if not os.path.exists("plots"):
    os.mkdir("plots")

for model in ["customized", "vgg16_pretrained", "vgg16_scratch"]:
    for metric in ["accuracy", "loss"]:
        show_training_history(model, metric)
