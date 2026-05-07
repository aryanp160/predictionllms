import pandas as pd
import numpy as numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import LSTM, LeakyReLU, CuDNNLSTM, CuDNNGRU
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import h5py
import os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import regularizers

tf.compat.v1.disable_eager_execution()


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=config))

with h5py.File(''.join(['data/allcoin2015to2017_wf.h5']), 'r') as hf:
    datas = hf['inputs'][()]
    labels = hf['outputs'][()]


nb_partitions = datas.shape[0]

step_size = datas.shape[2]
units= 50
second_units = 30
batch_size = 8
nb_features = datas.shape[3]
epochs = 50
output_size=1

output_file_name='allcoin2015to2017_WF_GRU_tanh_leaky'
#split training validation

for partition in range(nb_partitions):
    training_datas = datas[partition,:8000*4,:,:]
    validation_datas = datas[partition,8000*4:,:,:]
    training_labels = labels[partition,:8000*4,:,:]
    validation_labels = labels[partition,8000*4:,:,:]
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    csvlog = CSVLogger('result/'+output_file_name+'.csv', append=True)
    checkpoint = ModelCheckpoint('weights/'+output_file_name+'partition_'+str(partition)+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    cb_list = [early,csvlog,checkpoint]
    print(training_datas.shape, validation_datas.shape, training_labels.shape, validation_labels.shape)
    #build model
    model = Sequential()
    model.add(CuDNNGRU(units=units, input_shape=(step_size,nb_features),return_sequences=True))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=16))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=cb_list, verbose=0)

# model.fit(datas,labels)
#model.save(output_file_name+'.h5')


