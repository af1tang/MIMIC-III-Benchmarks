# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 17:49:11 2017

@author: andy
"""

import sys
import pandas as pd
import numpy as np
import pickle
import math, random
import datetime, time
import tensorflow as tf
import gensim
import matplotlib.pyplot as plt

#from pandas.tools.plotting import scatter_matrix
from scipy import stats
from time import sleep
from datetime import timedelta
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, mean_squared_error
from sklearn.utils import shuffle

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge, Activation, Dropout, TimeDistributed, Bidirectional, Masking
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.preprocessing import sequence
from keras import metrics

x_file = ''
y_file = ''
dx_file = ''
h5file = ''

def main(X,Y, dx):
    
    target_rep = False
    stateful = False
    bidirectional = True

    
    skf = KFold(n_splits=5, random_state = 8)
    
        
    #standardize X
    tmp = []
    for i in range(len(X)):
        if len(X[i]) < 24: 
            pass
        else:
            tmp.append(np.array(X[i][-24:]))

    tmp = np.array(tmp).T
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2]).T
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    del tmp
    
    #make patient seqs into 48 hr window. Average observation hours is 72.
    maxlen = 12
    for idx in range(len(X)):
        if len(X[idx]) > maxlen:
            #X[idx] = np.concatenate((X[idx][0:24], X[idx][-24:]), axis = 0)
            X[idx] = X[idx][0:12]
    X= sequence.pad_sequences (X, maxlen, dtype = 'float32')
    

    #format: data[count][loss/auc] [mean_tr, mean_te]
    data = {}
    
    start = time.time(); count = 0
    for train_index, test_index in skf.split(X, Y):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = lstm_model(input_shape = (X_train.shape[1], X_train.shape[2]), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)

        
        #if time distributed, must add new axes to Y's
        y_train = y_train.reshape(y_train.shape[0], 1)
        y_train = np.repeat(y_train[:, np.newaxis, :], maxlen, axis=1)
        for i in range(len(y_train)):
            for j in range(maxlen):
                y_train[i][j] = ((timedelta(days = y_train[i][j][0]) - timedelta(hours = j)))/timedelta(days=1)

        if (target_rep == False):
            y_train = y_train[:, -1, :]
        
        data[count] = {}
        data[count]['tr_mse'] = []
        data[count]['te_mse'] = []
                
        for epoch in range(30):
            #start = time.time()
            print ("Epoch # {0}:".format(epoch+1))
            if (stateful == True):
                model.reset_states()
                history = model.fit(X_train, y_train, nb_epoch = 1, shuffle = False, batch_size = 1)
                y_pred = model.predict(X_train, batch_size = 1)
                yhat = model.predict (X_test, batch_size = 1)

            else:
                X_train, y_train = shuffle(X_train, y_train, random_state =8)
                X_test, y_test = shuffle(X_test, y_test, random_state = 8)
                history = model.fit(X_train, y_train, nb_epoch = 1, shuffle = True, batch_size = 128)
                y_pred = model.predict(X_train, batch_size = 128)
                yhat = model.predict (X_test)
            
            if (target_rep == True):
                y_pred = y_pred[:, -1, :]
                y_train_last = y_train[:, -1, :]
                yhat = yhat[:, -1, :]

            tr_mse = mean_squared_error(y_train_last, y_pred)
            te_mse = mean_squared_error(y_test, yhat)
        
            data[count]['tr_mse'].append(tr_mse)
            data[count]['te_mse'].append(te_mse)
            
    model.save(h5file)
    
def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)
            
def lstm_model(input_shape, stateful = False, target_rep = False, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        if (stateful == True):
            model.add(Masking(mask_value = 0., batch_input_shape = (1, input_shape[0], input_shape[1])))
            model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful =stateful), merge_mode = 'concat'))
        else:
            model.add(Masking(mask_value=0., input_shape=input_shape))
            model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    else:
        if (stateful == True):
            model.add(Masking(mask_value = 0., batch_input_shape = (1, input_shape[0], input_shape[1])))
            model.add(LSTM(256, return_sequences=target_rep, stateful=stateful))
        else:
            model.add(Masking(mask_value=0., input_shape=input_shape))
            model.add(LSTM(256, return_sequences=target_rep, stateful=stateful))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))

    #if (bidirectional == True):
    #    model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    #else:
    #    model.add(LSTM(256, return_sequences = target_rep, stateful = stateful)) 
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        model.add(Dense(1))
    #model.add(BatchNormalization())
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return (model)

    
def plotting(loss, title, data, pkl):
    pkl = ''
    %matplotlib qt
    for idx in range(1, len(data)+1):
        title = "LOS prediction with raw LSTM MSE regression, KFOLD: {0}".format(idx)
        tr_mse = data[idx]['tr_mse'] 
        te_mse = data[idx]['te_mse']
        
        plt.subplot(len(data), 1, idx)
        plt.plot(tr_mse)
        plt.plot(te_mse)
        plt.title(title)
        plt.ylabel('performance')
        if (idx == len(data)):
            plt.xlabel('epoch', fontsize = 11)
        plt.legend(['tr_mse', 'te_mse'], loc='best')
        
        dat[idx] = {}
        dat[idx]['te_mse'] = data[idx]['te_mse'][-1]
    plt.show()
    
    #saving dataframe
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = ['KFOLD 1', 'KFOLD 2', 'KFOLD 3', 'KFOLD 4', 'KFOLD 5']
    df.to_pickle(pkl)
    

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    dx = np.load(dx_file)
     
    main(X, Y, [942, 0, 20]))

