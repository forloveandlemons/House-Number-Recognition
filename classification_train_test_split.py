
import os
import numpy as np
print("load train, test, extra folders")
c_train_data = np.load("alldata/classification/data/train.npy")
c_test_data = np.load("alldata/classification/data/test.npy")
c_extra_data = np.load("alldata/classification/data/extra.npy")

c_train_labels = np.load("alldata/classification/labels/train.npy")
c_test_labels = np.load("alldata/classification/labels/test.npy")
c_extra_labels = np.load("alldata/classification/labels/extra.npy")

print("run train, valiation data split")


full_train = np.concatenate([c_train_data, c_extra_data])
full_labels = np.concatenate([c_train_labels, c_extra_labels])

arr_rand = np.random.rand(len(c_train_labels) + len(c_extra_labels))
split = arr_rand < np.percentile(arr_rand, 75)

X_train = full_train[split]
X_valid = full_train[~split]
X_test = c_test_data

y_train = full_labels[split]
y_valid = full_labels[~split]
y_test = c_test_labels


if not os.path.exists("classification_data"):
        os.mkdir("classification_data")
for d in ["train", "valid", "test"]:
    if not os.path.exists("classification_data/{}".format(d)):
        os.mkdir("classification_data/{}".format(d))
    for type in ["data", "labels"]:
        if not os.path.exists("classification_data/{}/{}".format(d, type)):
            os.mkdir("classification_data/{}/{}".format(d, type))


np.save("classification_data/train/data", X_train)
np.save("classification_data/test/data", X_test)
np.save("classification_data/valid/data", X_valid)


np.save("classification_data/train/labels", y_train)
np.save("classification_data/test/labels", y_test)
np.save("classification_data/valid/labels", y_valid)
