# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:25:45 2017

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

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
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
sentence_file = ''
h5file = ''

def main(X,Y, dx):
    
    target_rep = False
    stateful = False
    bidirectional = True
    word2vec = False
    
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    
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
    #maxlen = 48
    #for idx in range(len(X)):
    #    if len(X[idx]) > maxlen:
            #X[idx] = np.concatenate((X[idx][0:24], X[idx][-24:]), axis = 0)
     #       X[idx] = X[idx][-48:]
    #X= sequence.pad_sequences (X, maxlen)

    
    #train diagnostic history
    if (word2vec):
        dx = np.ndarray.tolist(dx)
        SG = gensim.models.Word2Vec(sentences = dx, sg = 1, size = 24, window = 5, hs = 1, negative = 0)
        weights = SG.wv.syn0
        vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
        w2i, i2w = vocab_index(vocab)
        
        #turn sentences into word vectors for each admission
        dx = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv)) for vv in dx]    
        #word vectors here
        w2v = [] 
        
        for sentence in dx:
            one_hot = np.zeros((len(sentence), weights.shape[0]))
            one_hot[np.arange(len(sentence)), sentence] = 1
            one_hot = np.sum(one_hot, axis= 0)
            w2v.append(np.dot(one_hot.reshape(one_hot.shape[0]), weights))
            
        #standardize Word Vectors (optional)
        scaler = StandardScaler()
        w2v = scaler.fit_transform(w2v)
        
        #concatenate X with w2v
        #first, extend w2v to 3D tensor
        w2v = np.repeat(w2v[:, np.newaxis, :], maxlen, axis=1)
        #make new X
        X = np.concatenate((X, w2v), axis = 2)

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
        
        model = lstm_model(batch_input_shape = (1, 1, X_train.shape[2]), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['tr_matrix'] = []
        data[count]['tr_loss'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
                
        for epoch in range(30):
            #start = time.time()
            #X_train, y_train = shuffle(X_train, y_train, random_state =8)
            #X_test, y_test = shuffle(X_test, y_test, random_state = 8)
            y_pred = []; tr_loss = []; tr_acc = []
            for i in range(len(X_train)):
                sleep(.0001)
                y_true = y_train[i]
                timesteps = len(X_train[i])
                for j in range(timesteps):
                    loss, acc = model.train_on_batch(X_train[i][j].reshape(1, 1, X_train[i][j].shape[0]), y_true.reshape(1,5))
                    tr_acc.append(acc)                    
                    tr_loss.append(loss)                
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='* int((i+1)/(len(X_train)/20)), int(5*(i+1))/(len(X_train)/20)))
                sys.stdout.write(', Acc = %.3f' % (np.mean(tr_acc)))
                sys.stdout.write(', Loss = %.3f' % (np.mean(tr_loss)))
                sys.stdout.flush()
            model.reset_states()
            
            for i in range(len(X_test)):
                y_true = y_test[i]                
                timesteps = len(X_test[i])                
                for j in range(timesteps):
                    yhat = model.predict_on_batch(X_test[i][j].reshape(1, 1, X_test[i][j].shape[0]))
                y_pred.append(yhat.reshape(y_true.shape))
            model.reset_states()    
            
            y_pred = np.array(y_pred)
            fpr = dict()
            tpr = dict()
            #accuracy = dict()
            roc_auc = dict()
            for idx in range(Y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], y_pred[:, idx])
                roc_auc[idx] = auc(fpr[idx], tpr[idx])
                #accuracy[idx] = accuracy_score(y_test[:, idx], [round(yy) for yy in y_pred[:, idx]])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])     
            
            print('EPOCH {0}'.format(epoch))
            print('AUROC (Readmission): {0}'.format(roc_auc[1]))
            print('AUROC (Hospital Mort): {0}'.format(roc_auc[2]))
            print('AUROC (30d Mort): {0}'.format(roc_auc[3]))
            print('AUROC (1yr Mort): {0}'.format(roc_auc[4]))
            print('AUROC (Micro) =  {0}'.format(roc_auc['micro']))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
            data[count]['ACC'].append(np.mean(tr_acc))
            data[count]['AUC'].append(roc_auc)
            data[count]['LOSS'].append(np.mean(tr_loss))

            
def lstm_model(batch_shape = (1, 1, 19), dropout_W = 0.5, dropout_U = 0.2, optimizer = 'adam', neurons = 100, learn_rate = 1e-3, W_regularizer = None, U_regularizer = None, init_mode = 'glorot_uniform'):
    model = Sequential()
    model.add(LSTM(100, batch_input_shape= batch_shape, return_sequences=True, stateful=True))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(LSTM(100, batch_input_shape=batch_shape, return_sequences = False, stateful = True))    
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(5, init = 'glorot_uniform'))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy])
    return (model)

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    
    main(X, Y)