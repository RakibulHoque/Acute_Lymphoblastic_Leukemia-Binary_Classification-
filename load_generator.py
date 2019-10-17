from keras.utils import to_categorical
from imageio import imread
import numpy as np
import random
#random.seed(7)

from observe_data import all_imgs_dict, hem_imgs_dict
from augmentation import transform

#making two classes equal in training data
keys = list(all_imgs_dict.keys())
random.shuffle(keys)
all_img_dict_trimmed = {x:all_imgs_dict[x] for x in keys[0:len(hem_imgs_dict)]}
#all_img_dict_trimmed = all_imgs_dict
#modify ALL data
keys = list(all_img_dict_trimmed.keys())
random.shuffle(keys)
all_train_data = {x:all_img_dict_trimmed[x] for x in keys[0:4*(len(keys)//5)]}
all_valid_data = {x:all_img_dict_trimmed[x] for x in keys[4*(len(keys)//5):]}
#modify healthy data
keys = list(hem_imgs_dict.keys())
random.shuffle(keys)
hem_train_data = {x:hem_imgs_dict[x] for x in keys[0:4*(len(keys)//5)]}
hem_valid_data = {x:hem_imgs_dict[x] for x in keys[4*(len(keys)//5):]}
#training data final 

train_data = {**all_train_data,**hem_train_data}
valid_data = {**all_valid_data,**hem_valid_data}


'''extra data where train and valid is not equally distributed.'''
#data = {**all_img_dict_trimmed,**hem_imgs_dict} 
#generator for training
def generator_for_dict(data, img_size=(450,450,3), num_class=2, batchsize = 32, load_augmentor = False):
    keys = list(data.keys())
    img_rows, img_cols, channel = img_size
    batch_x = np.zeros((batchsize, img_rows, img_cols, channel)) 
    batch_y = np.zeros((batchsize, num_class)) 
    while 1:
        random.shuffle(keys)
        for i_key in range(0,len(keys) - len(keys)%batchsize, batchsize):
            for i_batch in range(batchsize):
                packet = data[keys[i_key+i_batch]]
                x = imread(packet['PATH'])
                if load_augmentor:
                    x = transform(x)
                y = to_categorical(packet['CONDITION'], num_class)
                batch_x[i_batch] = x
                batch_y[i_batch] = y
        yield batch_x, batch_y


"""loading from RAM"""
#def generator_for_dict(data, img_size=(450,450,3), num_class=2, batchsize = 32, load_augmentor = False):
#    keys = list(data.keys())
#    img_rows, img_cols, channel = img_size
#    batch_x = np.zeros((batchsize, img_rows, img_cols, channel)) 
#    batch_y = np.zeros((batchsize, num_class)) 
#    while 1:
#        random.shuffle(keys)
#        data_in_ram = list(map(imread,[packet['PATH'] for packet in 
#                                       [data[p] for p in keys]]))
#        label_in_ram = [packet['CONDITION'] for packet in 
#                                       [data[p] for p in keys]]
#        for i_key in range(0,len(data_in_ram) - len(data_in_ram)%batchsize, batchsize):
#            for i_batch in range(batchsize):
#                x = data_in_ram[i_key + i_batch]
#                if load_augmentor:
#                    x = transform(x)
#                y = to_categorical(label_in_ram[i_key + i_batch], num_class)
#                batch_x[i_batch] = x
#                batch_y[i_batch] = y
#        yield batch_x, batch_y   
