from load_generator import train_data, valid_data, generator_for_dict
from network import Network
from keras.optimizers import Adam
#from keras.optimizers import SGD
#hyperparameters
num_class = 2
img_size = (450,450,3)
nb_epoch = 100
batchsize = 32
weight_sv_dir = "new_custom_network_hope2.h5"
training_gen = generator_for_dict(train_data,
                                  img_size = img_size, 
                                  batchsize = batchsize, 
                                  num_class = num_class, 
                                  load_augmentor=True)
val_gen = generator_for_dict(valid_data,
                             img_size = img_size, 
                             batchsize = batchsize, 
                             num_class = num_class, 
                             load_augmentor=False)

model = Network(img_size,num_class)
model.load_weights("new_custom_network_hope_epoch_65.h5")
#%%
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def scheduler(epoch):

    if epoch!=0 and epoch%10 == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.5)
        print("lr changed to {}".format(lr*.5))

    return K.get_value(model.optimizer.lr)


def learning_rate_scheduler(filepath_name):
    lr_decay = LearningRateScheduler(scheduler)
    callbacks_list= [
        ModelCheckpoint(
            filepath= filepath_name,
            mode='min',
            monitor='val_loss',
            save_weights_only = False,
            save_best_only=True,
            verbose = 1
        ), lr_decay]
    return callbacks_list

callbacks_list = learning_rate_scheduler(weight_sv_dir)
#%%

#sgd = SGD(lr=0.0005, 
#          momentum=0.9,  
#          decay=7.5e-3, 
#          nesterov=False)
adam = Adam(lr=1.5625e-06, 
            beta_1=0.9, 
            beta_2=0.999)

model.compile(optimizer=adam, 
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(training_gen,
                    steps_per_epoch=int(len(train_data)/batchsize),
                    nb_epoch=nb_epoch,
                    validation_data=val_gen,
                    validation_steps=int(len(train_data)/batchsize),
                    callbacks=callbacks_list,
                    initial_epoch=65)
