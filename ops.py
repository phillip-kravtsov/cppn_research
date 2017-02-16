from scipy import ndimage
from scipy import misc
import imageio
from math import sqrt
from functools import reduce
import numpy as np


def pull_image(file_name):
    image = misc.imread(file_name)
    image = image * 1.0/255
    print ("size pulled = ", image.shape)
    return image
    
def get_data(image, color_string='RGB'):
    #image = misc.imread(file_name, mode=color_string)
    #image = misc.imresize(image, resize_perc)
    #image = image * 1.0/255
    
    positions = np.empty([image.shape[0] * image.shape[1], 2], dtype=np.float32)
    if (color_string == 'L'):
        pixel_values = np.empty([image.shape[0] * image.shape[1], 1], dtype=np.float32)
    elif (color_string == 'RGB'):
        pixel_values = np.empty([image.shape[0] * image.shape[1], 3], dtype=np.float32)
    
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if (color_string == 'L'):
                pixel_values[x * image.shape[1] + y, 0] = image[x,y]
            elif (color_string == 'RGB'):
                pixel_values[x * image.shape[1] + y, 0:3] = image[x,y,0:3]
            
            positions[x * image.shape[1] + y, 0] = 2.0 * float(x)/image.shape[0] - 1.0 
            positions[x * image.shape[1] + y, 1] = 2.0 * float(y)/image.shape[1] - 1.0
    
    return (positions, pixel_values, image.shape[0], image.shape[1])
def mix(data, n):
    ret = []
    for i in range(n):
        tiled = np.tile(np.eye(n)[i], (data[i].shape[0], 1))
        ret.append(np.concatenate([data[i], tiled], axis=1))
    return ret
def scale(x, size):
    return 1.00 * (2.0 * x / size - 1.0)
def prepare_data(file_names, perc, auto_size=False):
    n = len(file_names)
    images = [pull_image(x) for x in file_names]
    min_size = min(x.size for x in images)
    print ("Minimum size is: ", min_size)
    r_images = [1.0/255 * misc.imresize(x, perc * sqrt(min_size/x.size)) for x in images]
    
    dataset = []
    for i in range(n):
        dataset.append(get_data(r_images[i]))
    combined_x = [x[0] for x in dataset]
    mixed = mix(combined_x, n=n)
    x = np.concatenate(mixed)
    y = np.concatenate([x[1] for x in dataset])
    print (x.shape)
    print (y.shape)
    return (x,y, dataset[0][2], dataset[0][3])
            
def construct_image(model, x_size, y_size, z=np.array([1])):
    final = np.empty((x_size,y_size,3))
    for i in range(x_size):
        x = np.empty((y_size,2 + z.size))
        x[:,0] = scale(i, x_size)
        x[:,1] = scale(np.arange(y_size), y_size)
        x[:,2:] = z    
        final[i,:] = model.predict(x, batch_size=y_size)
    return final

def get_image(array, file_name):
    misc.imsave(file_name, array)
    print("saved")
