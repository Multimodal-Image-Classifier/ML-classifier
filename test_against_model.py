import pickle
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array,smart_resize

def format(img_to_input):
    img = smart_resize(img_to_input,(300,300))
    return img

img_label = {"car":0,
             "standing":1}

model = pickle.load(open('naked_data_small.h5'))
img = load_img('original/IMG_3505.JPG')


