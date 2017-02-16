from scipy import misc
import numpy as np
import keras.callbacks
import ops

class WriteImage(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs=None):
        for i in range(self.z.shape[0]):
            print ("z[i] is:", self.z[i])
            image = ops.construct_image(self.model, self.x_size, self.y_size, z=self.z[i])
            ops.get_image(image, file_name='Iter/e_' + str(epochs) + '_' + str(i) + '.png')
    def __init__(self, x_size, y_size, z):
        self.x_size = x_size
        self.y_size = y_size
        self.z = z
