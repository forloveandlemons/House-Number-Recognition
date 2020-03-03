import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import h5py
import urllib
import pickle
import cv2



def download(filename):
    url = 'http://ufldl.stanford.edu/housenumbers/'
    if not os.path.exists(filename):
        print('Downloading:', filename) 
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
        print('Completed')
    return filename


np.random.seed(133)

def extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root):
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
    data_folders = root
    print(data_folders)
    return data_folders

class DecoderWrapper:
    # referring to https://github.com/hangyao/street_view_house_numbers/blob/master/1_preprocess_single.ipynb
    def __init__(self, inf):
        self.f = h5py.File(inf, 'r')
        self.box = self.f['digitStruct']['bbox']
    
    def helper(self, attr):
        if (len(attr) > 1):
            attr = [self.f[attr[j][0]][0][0] for j in range(len(attr))]
        else:
            attr = [attr[0][0]]
        return attr

    def get_box(self, i):
        res = {}
        box_idx = self.f['digitStruct']['bbox'][i].item()
        res['height'] = self.helper(self.f[box_idx]["height"])
        res['label'] = self.helper(self.f[box_idx]["label"])
        res['left'] = self.helper(self.f[box_idx]["left"])
        res['top'] = self.helper(self.f[box_idx]["top"])
        res['width'] = self.helper(self.f[box_idx]["width"])
        return res
    
    def get_name(self, i):
        return ''.join([chr(c[0]) for c in self.f[self.f['digitStruct']['name'][i][0]].value])
    
    def get_metadata(self):
        print("get metadata")
        result = []
        for i in range(len(self.f['digitStruct']['name'])):
            if (i % 10000 == 0):
                print(i) 
            image_data = self.get_box(i)
            figures = []
            item = {"filename": self.get_name(i)}
            for j in range(len(image_data["height"])):
                temp = [image_data["label"][j],
                        image_data["height"][j], 
                        image_data["left"][j], 
                        image_data["top"][j], 
                        image_data["width"][j]]
                figures.append(temp)
            item['boxes'] = figures
            result.append(item)
        return result


def get_locations(left, top, width, height, img):
    min_top = np.max([0, np.int16(np.min(top))])
    min_left = np.max([0, np.int16(np.min(left))])
    max_bottom = np.int16(np.max(height) + min_top)
    max_right = np.int16(np.sum(width)  + min_left)
    height_padding = (max_bottom - min_top) * .3
    width_padding = (max_right - min_left) * .3
    min_top = np.max([0,  np.int16(min_top - height_padding)])
    max_bottom = np.min([img.shape[0], np.int16(max_bottom + height_padding)])
    min_left = np.max([0,  np.int16(min_left - width_padding)])
    max_right = np.min([img.shape[1], np.int16(max_right + width_padding)])
    return min_left, min_top, max_right - min_left, max_bottom - min_top

def get_detection_locations(x, y, w, h, detection_input_size, img):
    # get location of new box given resize scale
    new_x = int(x * detection_input_size / img.shape[1])
    new_y = int(y * detection_input_size / img.shape[0])
    new_w = int(w * detection_input_size / img.shape[1])
    new_h = int(h * detection_input_size / img.shape[0])
    return new_x, new_y, new_w, new_h    

def prepare_dataset(data, folder, classification_input_size, detection_input_size):
    print("generating classification & detection dataset")
    classification_input = np.ndarray([len(data), classification_input_size, classification_input_size,3], dtype='float32')
    classification_lables = np.ones([len(data),6], dtype=int) * 10
    detection_input = np.ndarray([len(data), detection_input_size, detection_input_size,3], dtype='float32')
    detection_labels = np.ones([len(data),4], dtype=int)
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = cv2.cvtColor(cv2.imread(fullname), cv2.COLOR_BGR2RGB)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        classification_lables[i,0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')  
        for j in range(num_digit):
            if j < 5:
                classification_lables[i,j+1] = boxes[j][0]
                if boxes[j][0] == 10:
                    classification_lables[i,j+1] = 0
            height[j] = boxes[j][1]
            left[j] = boxes[j][2]
            top[j] = boxes[j][3]
            width[j] = boxes[j][4]
        x, y, w, h = get_locations(left, top, width, height, im)
        new_x, new_y, new_w, new_h = get_detection_locations(x, y, w, h, detection_input_size, im)
        # get classification input image ~ crop at the outmost box corners
        im = im[y:y+h, x:x+w, :]
        classification_im = cv2.resize(im, (classification_input_size, classification_input_size))
        detection_im = cv2.resize(im, (detection_input_size, detection_input_size))
        classification_input[i, :, :, :] = classification_im[:, :, :]
        detection_input[i, :, :, :] = detection_im[:, :, :]
        detection_labels[i, :] = [new_x, new_y, new_w, new_h]
    return classification_input, classification_lables, detection_input, detection_labels


folder = sys.argv[1]
filename = maybe_download('{}.tar.gz'.format(folder))
folders = maybe_extract(filename)
data = DecoderWrapper("{}/digitStruct.mat".format(folder)).get_metadata()
c_data, c_labels, d_data, d_labels = prepare_dataset(data, folder, 32, 128)


if not os.path.exists("alldata"):
        os.mkdir("alldata")
for d in ["classification", "detection"]:
    if not os.path.exists("alldata/{}".format(d)):
        os.mkdir("alldata/{}".format(d))
    for type in ["data", "labels"]:
        if not os.path.exists("alldata/{}/{}".format(d, type)):
            os.mkdir("alldata/{}/{}".format(d, type))

print("saving classification input & labels")
np.save("alldata/classification/data/{}".format(folder), c_data)
np.save("alldata/classification/labels/{}".format(folder), c_labels)

if folder in ["train", "test"]:
    print("saving detection input & labels")
    np.save("alldata/detection/data/{}".format(folder), d_data)
    np.save("alldata/detection/labels/{}".format(folder), d_labels)




