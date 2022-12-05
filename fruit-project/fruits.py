# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:36:57 2022

@author: EBRU
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
import warnings
warnings.filterwarnings('ignore') 



train_path= "C://Users//EBRU//Desktop//deep learning//cnn-extra//fruits-360_dataset//fruits-360//Training"
test_path= "C://Users//EBRU//Desktop//deep learning//cnn-extra//fruits-360_dataset//fruits-360//Test"

img= load_img(train_path + "/Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x= img_to_array(img)
print(x.shape)

className= glob(train_path+'/*')
numberOfClass= len(className)
print("NumberOfClass: ",numberOfClass)

model= Sequential()
model.add(Conv2D(32, (3,3), input_shape= x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Dropout(0.5))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))


model.add(Dense(numberOfClass))
model.add(Activation("softmax"))

model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy",  metrics=["accuracy"])

batch_size = 32
a = glob(train_path + "/*")
len(a)


# data augmentation
train_datagen = ImageDataGenerator( 
        rescale = 1./255, 
        shear_range = 0.3, 
        zoom_range = 0.3,       
        horizontal_flip=True,   
            )

test_datagen = ImageDataGenerator(rescale = 1./255 )

train_generator = train_datagen.flow_from_directory(train_path, 
                                                    target_size = x.shape[:2],
                                                    batch_size = batch_size,
                                                    color_mode= "rgb", 
                                                    class_mode="categorical" 
                                                    )

test_generator = train_datagen.flow_from_directory  (test_path, 
                                                    target_size = x.shape[:2],
                                                    batch_size = batch_size,
                                                    color_mode= "rgb", 
                                                    class_mode="categorical"  
                                                    )

hist = model.fit_generator(
                    generator = train_generator,
                    steps_per_epoch = 1600 // batch_size,
                    epochs = 100,
                    validation_data = test_generator,
                    validation_steps = 800 // batch_size
                        )     


# saving the model
model.save_weights("fruit_hist.h5")

# evaluating the model
print(hist.history.keys())
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label=" Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label="Train accuracy")
plt.plot(hist.history["val_accuracy"], label= "Validation accuracy")
plt.legend()
plt.show()
# history'i yükleyebiliyoruz

#save history
import json
with open("fruit_hist.json","w") as f:
    json.dump(hist.history, f)
# history'i kaydedebiliyoruz
    
# uzun süren trainler sonucu elde ettiğimiz verileri depolayıp
# daha sonra tekrar train sonucu almadan kullanabilmek için depolama yapıyoruz.
 
import codecs
with codecs.open("fruit_hist.json", "r", encoding="utf-8") as f:
    h= json.loads(f.read())
plt.plot(h["loss"], label="Train Loss")
plt.plot(h["val_loss"], label=" Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label="Train accuracy")
plt.plot(h["val_accuracy"], label= "Validation accuracy")
plt.legend()
plt.show()

# daha önce 100 epoch'ta sonucunu aldığımız modeli tekrar yükleyip sonuçlara göz atabiliyoruz.
# history'i çizdirebiliyoruz
# bu son yazdığımız kod bloğu ile sonuç olarak aldığımız son iki grafiği aynı anda çizdirebildik.











