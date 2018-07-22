import os
from PIL import Image

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

def get_dataset(cur_dir):
    classes = os.listdir(cur_dir)
    image_dict = {class_:[] for class_ in classes}
    for class_ in classes:
        image_dir = os.path.join(cur_dir, class_)
        images = os.listdir(image_dir)
        for im in images:
            image_dict[class_].append(join_directories([cur_dir, image_dir, im]))
    return image_dict

def get_class_to_int(classes):
    ret = {}
    ret2 = {}
    for i, class_ in enumerate(sorted(list(classes))):
        ret[class_] = i
        ret2[i] = class_
    return ret, ret2

if __name__ == '__main__':
    training = get_dataset(os.path.join("dataset", "training"))
    validation = get_dataset(os.path.join("dataset", "validation"))
    joined_dataset = join_datasets(training, validation)
    class_to_int, int_to_class = get_class_to_int(joined_dataset.keys())
