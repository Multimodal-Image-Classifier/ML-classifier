import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation,Dropout, Flatten, Dense
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img

train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train = train_datagen.flow_from_directory('data/tran',
                                          batch_size=30,
                                           target_size=(300,300),
                                           class_mode= 'binary',
                                         shuffle = True)

validation = validation_datagen .flow_from_directory('data/validation',
                                          batch_size=30,
                                          target_size=(300,300),
                                          class_mode= 'binary',
                                          shuffle = True)


model =  keras.Sequential()


model.add(Conv2D(32,(3,3),input_shape=(300,300,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation ='softmax'))

model.compile(optimizer='adam',
              loss= losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
'''
history = model.fit(train_x, train_y, epochs=30 ,batch_size= 32
                    ,validation_data=(test_x, test_y))
'''

model.fit(
        train,
        steps_per_epoch=15, #batch_size,
        epochs=20,
        validation_data=validation,
        validation_steps=200) # batch_size)

pickle.dump(model,open('model_20_.001LR','wb'))
