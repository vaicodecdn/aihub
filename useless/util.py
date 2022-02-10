import math, os
from PIL import Image
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt

# convert image to grayscale
def to_one_channel(image):
    return image[:,:,0] if len(image.shape)>2 else image

# resize the image
def resize_image(image, new_shape):
    image = image.resize((new_shape),Image.ANTIALIAS)
    return image

# split dataset into train and test
def split_train_test(tooth, masks, train_percentage):
    divide_index = math.ceil(len(tooth)*train_percentage/100)
    X_train=tooth[:divide_index,:,:,:]
    X_test=tooth[divide_index:,:,:,:]
    y_train=masks[:divide_index,:,:,:]
    y_test=masks[divide_index:,:,:,:]
    return (X_train, y_train),(X_test, y_test)

# loading dataset function
def load_dataset(dataset_folder, rgb = False, normalize = True):
    images = []
    for teeth in natsorted(os.listdir(dataset_folder)):
        if teeth in os.listdir('./dataset/pbl-mask'):
            image = Image.open(os.path.join(dataset_folder, teeth))
            image = resize_image(image, (2752, 1372))
            width, height = image.size
            image = image.crop((300, 300, width-300, height))
            image = resize_image(image, (512, 512))
            image = np.asarray(image) if rgb else to_one_channel(np.asarray(image))
            images.append(image)
    if not rgb:
        images = np.asarray(images) / 255.0 if normalize else np.asarray(images)
        images = np.reshape(images, (len(images),images.shape[1], images.shape[2], 1))
    return images

# plot unet loss value
def unet_lost_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper right')
    plt.savefig('./result/unet/unet_perfomance.png', facecolor='white', dpi=1000)
    # plt.show()