# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:20:06 2017

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
from sklearn.preprocessing import Imputer, normalize, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils import shuffle, class_weight

from keras.preprocessing import sequence
#from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, merge, Activation, Dropout, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.preprocessing import sequence
from keras import metrics

x_file = ''
y_file = ''
h5file = ''

def main(X,Y):
    
    target_rep = True
    stateful = False
    bidirectional = True
    
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    
    #standardize X
    tmp = []
    for i in X:
        for j in i:
            tmp.append(j)
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    del tmp
    
    #make patient seqs into 48 hr window. Average observation hours is 72.
    maxlen = 48
    for idx in range(len(X)):
        if X[idx].shape[0] > maxlen:
            X[idx] = np.concatenate((X[idx][0:24], X[idx][-24:]), axis = 0)
    
    #for splitting purposes, use y: [0-5] for groups
    Y = Y.tolist()
    y = [l.index(1) for l in Y]
    Y = np.array(Y)
    
    #fix class imbalance w/ class weights
    class_wts = class_weight.compute_class_weight('balanced', np.unique(y), y)

    #format: data[count][loss/auc] [mean_tr, mean_te]
    data = {}
    
    start = time.time(); count = 0
    for train_index, test_index in skf.split(X, y):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        
        #if time distributed, must add new axes to Y's
        if (target_rep == True):
            y_train = np.repeat(y_train[:, np.newaxis, :], maxlen, axis=1)
                    
        X_train = sequence.pad_sequences (X_train, maxlen)
        X_test = sequence.pad_sequences (X_test, maxlen)
        
        model = lstm_model(input_shape = (X_train.shape[1], X_train.shape[2]), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)
        
        data[count] = {}
        data[count]['AUC'] = []
        data[count]['ACC'] = []
        data[count]['LOSS'] = []
                       
        for epoch in range(100):
            #X_train, y_train = shuffle(X_train, y_train, random_state =8)
            #X_test, y_test = shuffle(X_test, y_test, random_state = 8)
            #start = time.time()
            if (stateful == True):
                model.reset_states()
                history = model.fit(X_train, y_train, nb_epoch = 1, shuffle = False, batch_size = 1, class_weight = class_wts)
            else:
                history = model.fit(X_train, y_train, nb_epoch = 1, shuffle = True, batch_size = 128, class_weight = class_wts)
            y_pred = model.predict(X_test, batch_size = 1)
            
            if (target_rep == True):
                y_pred = y_pred[:, -1, :]
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for idx in range(Y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], y_pred[:, idx])
                roc_auc[idx] = auc(fpr[idx], tpr[idx])
                
            # Compute micro-average ROC curve and ROC area 
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            #print progress            
            print('EPOCH {0}'.format(epoch))
            print('AUROC (Readmission): {0}'.format(roc_auc[1]))
            print('AUROC (Hospital Mort): {0}'.format(roc_auc[2]))
            print('AUROC (30d Mort): {0}'.format(roc_auc[3]))
            print('AUROC (1yr Mort): {0}'.format(roc_auc[4]))
            print('AUROC (Micro) =  {0}'.format(roc_auc['micro']))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
        
            data[count]['AUC'].append(roc_auc)
            data[count]['ACC'].append(history.history['categorical_accuracy'])
            data[count]['LOSS'].append(history.history['loss'])
            
    model.save(h5file)

            
def lstm_model(input_shape, stateful = False, target_rep = True, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        if (stateful == True):
            model.add(Bidirectional(LSTM(100, batch_input_shape = (1, input_shape[0], input_shape[1]), return_sequences = True, stateful =stateful), merge_mode = 'concat'))
        else:
            model.add(Bidirectional(LSTM(100, input_shape = input_shape, return_sequences = True, stateful = stateful), merge_mode = 'concat'))
    else:
        if (stateful == True):
            model.add(LSTM(100, batch_input_shape = (1, input_shape[0], input_shape[1]), return_sequences=True, stateful=stateful))
        else:
            model.add(LSTM(100, input_shape = input_shape, return_sequences=True, stateful=stateful))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if (bidirectional == True):
        model.add(Bidirectional(LSTM(100, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    else:
        model.add(LSTM(100, return_sequences = target_rep, stateful = stateful)) 
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(5, activation = 'softmax')))
    else:
        model.add(Dense(5, activation = 'softmax'))
    #model.add(BatchNormalization())
    optimizer = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy])
    return (model)
    
def plotting(loss, title, data, pkl):
    auc = data['AUC']
    re = [i[1] for i in auc]
    mort_h = [i[2] for i in auc]
    mort_30 = [i[3] for i in auc]
    mort_1y = [i[4] for i in auc]
    micro = [i['micro'] for i in auc]
    loss = data['LOSS']
    loss = [i[0] for i in loss]
    acc = data['ACC']
    acc = [i[0] for i in acc]
    
    #%matplotlib qt
    plt.plot(loss)
    plt.plot(re)
    plt.plot(mort_h)
    plt.plot(mort_30)
    plt.plot(mort_1y)
    plt.plot(micro)
    plt.plot(acc)
    plt.title(title)
    plt.ylabel('performance')
    plt.xlabel('epoch')
    plt.legend(['loss', 'AUC readm', 'AUC mort_hosp', 'AUC mort_30d', 'AUC mort_1y', 'AUC micro', 'accuracy'], loc='upper right')
    plt.show()
    
    #saving dataframe
    dat = {}
    dat['AUC READMISSION'] = re
    dat['AUC MORT_H'] = mort_h
    dat['AUC MORT_30'] = mort_30
    dat['AUC MORT_1Y'] = mort_1y
    dat['AUC MICRO'] = micro
    dat['LOSS'] = loss
    dat['ACC'] = acc
    df = pd.DataFrame(dat)
    df.to_pickle(pkl + '.pkl')

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    
    main(X, Y)