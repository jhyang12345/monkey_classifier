import os
from PIL import Image
import numpy as np

im_size = 224

def join_directories(dirs):
    ret = ""
    for dir in dirs:
        ret = os.path.join(ret, dir)
    return ret

def join_datasets(training, validation):
    ret = {_class:[] for _class in training}
    for key in training:
        ret[key].extend(training[key])
    for key in validation:
        ret[key].extend(validation[key])
    return ret

def open_image_and_resize(image_directory, im_size=im_size):
    im = Image.open(image_directory)
    im = im.resize((im_size, im_size))
    im = np.array(im, dtype=np.float32)
    return im

def get_dataset(cur_dir, im_size=im_size):
    classes = os.listdir(cur_dir)
    image_dict = {class_:[] for class_ in classes}
    for class_ in classes:
        image_dir = os.path.join(cur_dir, class_)
        images = os.listdir(image_dir)
        for im in images:
            image_dict[class_].append(open_image_and_resize(join_directories([image_dir, im]), im_size))
    return image_dict

def get_class_to_int(classes):
    ret = {}
    ret2 = {}
    for i, class_ in enumerate(sorted(list(classes))):
        ret[class_] = i
        ret2[i] = class_
    return ret, ret2

def class_to_one_hot(class_, class_to_int):
    arr = [0. for _ in range(len(class_to_int.keys()))]
    arr[class_to_int[class_]] = 1.
    return arr

def get_input_datasets(im_size=im_size):
    training = get_dataset(os.path.join("dataset", "training"), im_size)
    validation = get_dataset(os.path.join("dataset", "validation"), im_size)
    class_to_int, int_to_class = get_class_to_int(training.keys())
    training_input = []
    training_output = []
    for class_ in training.keys():
        for image_data in training[class_]:
            training_input.append(image_data)
            training_output.append(class_to_one_hot(class_, class_to_int))
    training_input = np.asarray(training_input, dtype=np.float32)
    training_output = np.asarray(training_output, dtype=np.float32)
    validation_input = []
    validation_output = []
    for class_ in validation.keys():
        for image_data in validation[class_]:
            validation_input.append(image_data)
            validation_output.append(class_to_one_hot(class_, class_to_int))
    validation_input = np.asarray(validation_input, dtype=np.float32)
    validation_output = np.asarray(validation_output, dtype=np.float32)
    return training_input, training_output, validation_input, validation_output

if __name__ == '__main__':
    training = get_dataset(os.path.join("dataset", "training"))
    validation = get_dataset(os.path.join("dataset", "validation"))
    joined_dataset = join_datasets(training, validation)
    class_to_int, int_to_class = get_class_to_int(joined_dataset.keys())
    print(class_to_int, int_to_class)
    print(class_to_one_hot(list(class_to_int.keys())[0], class_to_int))
    get_input_datasets()
