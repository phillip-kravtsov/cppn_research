import numpy as np
import tensorflow as tf
import ops
import cb
import time
import imageio
from keras.models import Sequential
from keras.layers import Highway, Dense, Activation
from keras.callbacks import ModelCheckpoint

color_string = 'RGB'
file_names = ["Images/hr_sn.png", "Images/ml.jpg"]
n_ = len(file_names)

def add_layer(model, width, activation="tanh"):
    model.add(Highway(bias=True))
    #model.add(Dense(bias=True, output_dim=width))
    model.add(Activation(activation))
    return model

def scale(x, size):
    return 1.00 * (2.0 * x / size - 1.0)

model = Sequential()
model.add(Dense(output_dim=64, input_dim= 2 + n_, bias=True))
model.add(Activation("relu"))

for i in range(12):
    model = add_layer(model, 64, "relu")
model.add(Dense(output_dim=len(color_string), bias=True))
model.add(Activation('sigmoid'))
model.compile(optimizer="Adam", loss="binary_crossentropy")

data = ops.prepare_data(file_names,1.00)

x = data[0]
y = data[1]
x_size = 1000
y_size = int(1000 * data[3]/data[2])
c = [cb.WriteImage(x_size, y_size, z = np.eye(n_))]
model.fit(x, y, nb_epoch=50,  batch_size=32)

#ops.get_image(ops.construct_image(model, x_size, y_size, z = np.array([1.0,1.0, 1.0])) * 255, file_name='final.png')        
#ops.get_image(construct_image(np.array([0.0,1.0])) * 255, file_name='first.png')
#ops.get_image(construct_image(np.array([1.0,0.0])) * 255, file_name='second.png')
#ops.get_image(construct_image(np.array([2.0,0.0])) * 255, file_name='very_second.png')
a = np.linspace(0.0,1.0,num=50)
b = 1.0 - a
images = []
for i in range(50):
    images.append(ops.construct_image(model, x_size, y_size, z=np.array([a[i], b[i]])))
imageio.mimsave("end.gif", images)
