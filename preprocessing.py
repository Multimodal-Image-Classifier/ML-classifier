from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
import numpy as np

print('loading done')

batch_size = 20
train_datagen = ImageDataGenerator(rotation_range= 90,
                             width_shift_range=0.2,
                             height_shift_range = .2,
                             shear_range =0.2,
                             zoom_range=0.2,
                             horizontal_flip = True,
                             fill_mode='nearest',
                             rescale=1./255)

'''
 fuck my life
'''


'''
train = train_datagen.flow_from_directory('original/train',
                                          batch_size=400,
                                           target_size=(300,300),
                                           class_mode= 'binary',
                                         shuffle = True)

i = 0
for batch in train_datagen.flow_from_directory('original/train',save_to_dir='data/tran',
                                           target_size=(300,300),
                                           class_mode= 'categorical',
                                          shuffle = True):
    i+=1
    if i >30:
        break
'''
i = 0
for batch in train_datagen.flow_from_directory('original/validation',save_to_dir='data/validation',
                                           target_size=(300,300),
                                           class_mode= 'categorical',
                                          shuffle = True):
    i+=1
    if i >20:
        break




