# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:58:33 2017

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
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
from sklearn.utils import shuffle, class_weight

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
from keras import backend as k

x_file = ''
y_file = ''

h5file = ''

def main(X,Y, dx):
    
    target_rep = False
    stateful = False
    bidirectional = True
    mode = 'binary'
    hierarchal = False
    
    #if mortality: use 'post' (removes from posterior seq), if readm use 'pre' (uses posterior of seq)
    task = -1
    if (task == 2):
        truncate = 'post'
    else:
        truncate = 'pre'
    #Y is shaped: < time-to-event t, censor 0/1, mort_h 0/1, mort_30 0/1, readm 0/1 > 
    #mort_30 is only used for exclusion/inclusion purposes.     
    
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    
    #standardize X
    tmp = []
    maxlen = []
    for i in range(len(X)):
        if len(X[i]) < 24: 
            pass
        else:
            tmp.append(np.array(X[i][-24:]))
        maxlen.append(len(X[i]))

    tmp = np.array(tmp).T
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[1] * tmp.shape[2]).T
    scaler = StandardScaler()
    scaler = scaler.fit(tmp)
    for i in range(len(X)):
        X[i] = scaler.transform(X[i])
    del tmp
    
    #make patient seqs into 48 hr window. Average observation hours is 72.
    maxlen = 48
    X= sequence.pad_sequences (X, maxlen, dtype = 'float32', padding='pre', truncating = truncate)
    

    #format: data[count][loss/auc] [mean_tr, mean_te]
    data = {}
    
    start = time.time(); count = 0
    for train_index, test_index in skf.split(X, Y[:, task]):
        
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        if task == -1:
            #for readmission, remove mort_h and mort_30 from testing
            X_test, y_test = remove_subsample (X_test, y_test)
            X_train, y_train = modify_subsample (X_train, y_train)
        y_test = y_test[:, task]
            
        if mode == 'binary':        
            model = lstm_classifier(input_shape = (X_train.shape[1], X_train.shape[2]), target_rep = target_rep, bidirectional = bidirectional)

        elif mode == 'mse':
            model = lstm_regression(input_shape = (X_train.shape[1], X_train.shape[2]), target_rep = target_rep, bidirectional = bidirectional)           
            xs, ys = regression_subsample(X_train, y_train)
            ys = np.array([[c[0], c[4]] for c in ys])
            
        elif mode == 'survival':
            model = lstm_survival(input_shape = (X_train.shape[1], X_train.shape[2]), target_rep = target_rep, bidirectional = bidirectional)
            xs = X_train
            ys = np.array([[c[0], c[1], c[4]] for c in y_train])
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['tr_loss'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
                
        for epoch in range(30):
            X_train, y_train = shuffle(X_train, y_train, random_state =8)
            X_test, y_test = shuffle(X_test, y_test, random_state = 8)
            
            if target_rep:
                ys = np.repeat(ys[:, np.newaxis, :], maxlen, axis=1)
                for i in range(len(ys)):
                    for j in range(0, maxlen):
                        ys[i][j][0] = ((timedelta(days = ys[i][j][0]) - timedelta(hours = maxlen - j-1)))/timedelta(days=1)
        
            if mode == 'binary':
                ys = y_train[:, task]
                xs = X_train
                #downsample the training set
                xs, ys = balanced_subsample(xs, ys, 1.0)
                ys = np.array([[i] for i in ys])

                history = model.fit(xs, ys, nb_epoch = 1, shuffle = True, batch_size = 128)
                y_pred = model.predict(xs, batch_size = 128)
                yhat = model.predict (X_test, batch_size = 128)
                
                fpr, tpr, _ = roc_curve(ys, y_pred)
                tr_roc_auc = auc(fpr, tpr)
                matrix = confusion_matrix(y_test, np.array([round(i[0]) for i in yhat]))
                f1 = f1_score(y_test, np.array([round(i[0]) for i in yhat]))
                
            elif mode == 'mse':
                history = model.fit(xs, ys[:, 0], nb_epoch = 1, shuffle = True, batch_size = 128)
                y_pred = model.predict(xs, batch_size = 128)
                yhat = model.predict (X_test, batch_size = 128)
                y_pred = [(lambda t: 1 if t<=30 else 0)(t) for t in y_pred]
                yhat = [(lambda t: 1 if t<=30 else 0)(t) for t in yhat]
                
                fpr, tpr, _ = roc_curve(ys[:,1], y_pred)
                tr_roc_auc = auc(fpr, tpr)            
                matrix = confusion_matrix(y_test, yhat)
                f1 = f1_score(y_test, yhat)
                
            else:
                history = model.fit(xs, ys[:, 0:2], nb_epoch = 1, shuffle = True, batch_size = 128)
                y_pred = model.predict(xs, batch_size = 128)
                yhat = model.predict (X_test, batch_size = 128)
                y_pred = [(lambda t: 1 if t<=30 else 0)(t) for t in y_pred[:, 0]]
                yhat = [(lambda t: 1 if t<=30 else 0)(t) for t in yhat[:, 0]]
                
                fpr, tpr, _ = roc_curve(ys[:, 2], y_pred)
                tr_roc_auc = auc(fpr, tpr)
                matrix = confusion_matrix(y_test, yhat)
                f1 = f1_score(y_test, yhat)
            
            if (target_rep == True):
                y_pred = y_pred[:, -1, :]
                ys = ys[:, -1, :]
                yhat[:, -1, :]
            
            data[count]['tr_auc'].append(tr_roc_auc)
            data[count]['tr_loss'].append(history.history['loss'])
            
            fpr, tpr, _ = roc_curve(y_test, yhat)
            roc_auc = auc(fpr, tpr)
        
            data[count]['f1_score'].append(f1)
            data[count]['te_matrix'].append(matrix)
            data[count]['te_auc'].append(roc_auc)
            
            #print progress            
            print('Epoch {0}'.format(epoch))
            print('AUROC (training) =  {0}'.format(tr_roc_auc))
            print('AUROC (testing) = {0}'.format(te_roc_auc))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
            
    model.save(h5file)

def weibull_loglik_discrete(y_true, ab_pred, name=None):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]

    hazard0 = k.pow((y_ + 1e-35) / a_, b_)
    hazard1 = k.pow((y_ + 1) / a_, b_)

    return -1 * k.mean(u_ * k.log(k.exp(hazard1 - hazard0) - 1.0) - hazard1)

def activate(ab):
    a = k.exp(ab[:, 0])
    b = k.softplus(ab[:, 1])

    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))

    return k.concatenate((a, b), axis=1)
    

def lstm_classifier(input_shape, target_rep = False, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep), merge_mode = 'concat'))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(LSTM(256, return_sequences=target_rep))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        model.add(Dense(1, activation = 'sigmoid'))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)
    
def lstm_regression(input_shape, target_rep = False, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep), merge_mode = 'concat'))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(LSTM(256, return_sequences=target_rep))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        model.add(Dense(1))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return (model)

def lstm_survival(input_shape, target_rep = False, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(Bidirectional(LSTM(256, return_sequences = target_rep), merge_mode = 'concat'))
    else:
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(LSTM(256, return_sequences=target_rep))
    model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(1, activation = 'sigmoid')))
    else:
        model.add(Dense(2))
    model.add(Activation(activate))
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss=weibull_loglik_discrete, optimizer=optimizer)
    return (model)


def multitask_lstm (input_shape, embedding, stateful= False, target_rep = False, bidirectional = False):
    x = Input(shape = input_shape, name = 'x')
    dx = Input(shape = embedding, name = 'dx')

    xx = Bidirectional(LSTM (100, return_sequences = True, stateful = stateful, activation = 'relu'), merge_mode = 'concat', input_shape = (input_shape[0], input_shape[1])) (x)
    xx = Bidirectional(LSTM (100, return_sequences = target_rep, stateful = stateful, activation = 'relu'), merge_mode = 'concat') (xx)
    xx = Dropout(0.5)(xx)

    xx = merge([xx, dx], mode = 'concat')
    xx = Dense(64, activation = 'relu') (xx)
    y = Dense(1, activation = 'sigmoid') (xx)
    model = Model(input = [x, dx], output = [y])
    model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)
    
def plotting(loss, title, data, pkl):
    #title = "Stateful BILSTM on Mortality"
    #pkl = '/home/andy/Desktop/MIMIC/dat/baselines/readm/bilstm_binary.pkl'
    %matplotlib qt
    dat = {}
    te_matrix = []; te_auc = []
    pkl = '/home/andy/Desktop/MIMIC/dat/baselines/mortality/bilstm.pkl'
    for idx in range(1, len(data)+1):
        title = "Baseline Mortality Prediction w/ BILSTM, KFOLD: {0}".format(idx)
        tr_auc = data[idx]['tr_auc']
        tr_auc = [i for i in tr_auc]
        tr_loss = data[idx]['tr_loss']
        tr_loss = [i[0] for i in tr_loss]
        te_auc = data[idx]['te_auc']
        te_auc = [i for i in te_auc]
    
        plt.subplot(len(data), 1, idx)
        plt.plot(tr_loss)
        plt.plot(tr_auc)
        plt.plot(te_auc)
        plt.title(title)
        plt.ylabel('performance')
        if (idx == len(data)):
            plt.xlabel('epoch')
        plt.legend(['tr_loss', 'tr_auc', 'te_auc'], loc='best')
    
        dat[idx] = {}
        mat = data[idx]['te_matrix'][-1]
        tn, fp, fn, tp = mat.ravel()
        precision = 1.0 * (tp/(tp+fp))
        recall = 1.0* (tp/(tp+fn))
        spec = 1.0* (tn/(tn+fp))
        f1 = (2.0*precision * recall) / (precision + recall)
        dat[idx]['mat'] = data[idx]['te_matrix'][-1]
        dat[idx]['prec'] = precision
        dat[idx]['sen'] = recall
        dat[idx]['spec'] = spec
        dat[idx]['f1'] = f1
        dat[idx]['auc'] = data[idx]['te_auc'][-1]
    plt.show()
    
    #saving dataframe
    df = pd.DataFrame(dat)
    df = df.transpose()
    df.index.name = 'KFOLD'
    #df.to_pickle(pkl)
    #df.read_pickle(pkl) to test.    

def modify_subsample(x, y):
    for idx in range(len(y_train)):
        x[idx] = x[idx]
        if ((y[idx][2] == 1) | (y[idx][3] == 1) | (y[idx][4] == 1)):
            y[idx][4] =1
    return (np.array(x), np.array(y))

def remove_subsample(x, y):
    xt = []; yt = []
    for idx in range(len(x)):
        if ((y[idx][2] == 0) & (y[idx][3] == 0)):
            xt.append(x[idx])
            yt.append(y[idx])
    return (np.array(xt), np.array(yt))

def regression_subsample(x, y):
    xs = []; ys = []
    for idx in range(len(y_train)):
        if y[idx][0] != 720.:
            xs.append(x[idx])
            ys.append(y[idx][0:2])
    ys = np.array(ys)
    return (np.array(xs), np.array(ys))
    
def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    dx = np.load(dx_file)
     
    main(X, Y, [942, 0, 20]))

