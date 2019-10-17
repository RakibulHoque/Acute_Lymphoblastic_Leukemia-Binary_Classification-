from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D , AveragePooling2D, Dropout
from keras.engine.topology import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization

def Network(inp_size, num_class):
    inp = Input(shape = inp_size)
    
    x = Conv2D(16, (3, 3), strides = 1, input_shape = inp_size, padding='valid')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(32, (3, 3), strides = (1,1), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(48, (3, 3), strides = (1,1), padding='valid', bias_initializer = 'constant')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Conv2D(64, (5, 5), strides = (1,1), bias_initializer = 'constant', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid', data_format=None)(x)

    x = Flatten()(x)
    
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation = 'relu')(x)
    
    z = Dense(num_class, activation = 'softmax')(x)
    
    model = Model(inputs = inp, outputs= z)
    model.summary()
    return model

    
    

if __name__ == "__main__":
#hyperparameters
    num_class = 2
    img_size = (450,450,3)
    nb_epoch = 100
    batchsize = 32
    model = Network(img_size,num_class)
    model_json  = model.to_json()
    with open("model_new.json", "w") as json_file:
        json_file.write(model_json)