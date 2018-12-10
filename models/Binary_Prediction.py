# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:41:20 2017

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

x_file = ''
y_file = ''
dx_file = ''
h5file = ''

def main(X,Y, dx):
    
    target_rep = False
    stateful = False
    bidirectional = True
    word2vec = False
    hierarchal = False
    
    task = 0
    
    skf = StratifiedKFold(n_splits=5, random_state = 8)
    
    Y = Y[:, task]
    y = np.array([[i] for i in Y])
        
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
        for i in range(len(X)):
            tmp = w2v[i].reshape(1, len(w2v[i]))
            tmp = np.repeat(tmp, len(X[i]), axis = 0)
            X[i] = np.concatenate((X[i], tmp), axis= 1)
        
    #make patient seqs into 48 hr window. Average observation hours is 72.    
    maxlen = 48
    X= sequence.pad_sequences (X, maxlen, dtype = 'float32', padding='pre', truncating = 'pre', masking = 0.0)

    #format: data[count][loss/auc] [mean_tr, mean_te]
    data = {}
    
    start = time.time(); count = 0
    for train_index, test_index in skf.split(X, Y):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        xs, ys = balanced_subsample(X_train, [i[0] for i in y_train], 1.0)
        ys = np.array([[i] for i in ys])
        
        if (hierarchal):
            w2v_train = xs[:, 0, -24:]
            w2v_test = X_test [:, 0, -24:]
            xs = xs[:, :, 0:19]
            X_test = X_test[:, :, 0:19]
            model = hierarchal_lstm(input_shape = (xs.shape[1], xs.shape[2]), embedding = (w2v_train.shape[1],), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)
            
        else:
            model = lstm_model(input_shape = (xs.shape[1], xs.shape[2]), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)

        
        #if time distributed, must add new axes to Y's
        if (target_rep == True):
            ys = np.repeat(ys[:, np.newaxis, :], maxlen, axis=1)
        
        
        data[count] = {}
        data[count]['tr_matrix'] = []
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['tr_loss'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
                
        for epoch in range(30):
            #start = time.time()
            
            if (stateful == True):
                model.reset_states()
                history = model.fit(xs, ys, nb_epoch = 1, shuffle = False, batch_size = 1)
                y_pred = model.predict(xs, batch_size = 1)
                yhat = model.predict (X_test, batch_size = 1)

            else:
                xs, ys = shuffle(xs, ys, random_state =8)
                X_test, y_test = shuffle(X_test, y_test, random_state = 8)
                if (hierarchal):
                    history = model.fit(x = [xs, w2v_train], y = ys, nb_epoch = 1, shuffle = True, batch_size = 128)
                    y_pred = model.predict([xs, w2v_train], batch_size = 128)
                    yhat = model.predict ([X_test, w2v_test], batch_size = 128)
                else:
                    history = model.fit(xs, ys, nb_epoch = 1, shuffle = True, batch_size = 128)
                    y_pred = model.predict(xs, batch_size = 128)
                    yhat = model.predict (X_test)
            
            if (target_rep == True):
                y_pred = y_pred[:, -1, :]
                ys = ys[:, -1, :]
                yhat[:, -1, :]

            fpr, tpr, _ = roc_curve(ys, y_pred)
            tr_roc_auc = auc(fpr, tpr)
                            
            #print progress            
            print('Epoch {0}'.format(epoch))
            print('AUROC =  {0}'.format(tr_roc_auc))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
        
            data[count]['tr_auc'].append(tr_roc_auc)
            data[count]['tr_matrix'].append(tr_matrix)
            data[count]['tr_loss'].append(history.history['loss'])
            
            fpr, tpr, _ = roc_curve(y_test, yhat)
            roc_auc = auc(fpr, tpr)
            matrix = confusion_matrix(y_test, np.array([round(i[0]) for i in yhat]))
            f1_score = f1_score(y_test, np.array([round(i[0]) for i in yhat]))
        
            data[count]['f1_score'].append(f1_score)
            data[count]['te_matrix'].append(matrix)
            data[count]['te_auc'].append(roc_auc)
            
    model.save(h5file)
    
def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)
            
def lstm_model(input_shape, stateful = False, target_rep = False, bidirectional = False):
    model = Sequential()
    if (bidirectional == True):
        if (stateful == True):
            model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful =stateful), merge_mode = 'concat', batch_input_shape = (1, input_shape[0], input_shape[1])))
        else:
            model.add(Masking(mask_value=0., input_shape=input_shape))
            model.add(Bidirectional(LSTM(256, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    else:
        if (stateful == True):
            model.add(LSTM(256, batch_input_shape = (1, input_shape[0], input_shape[1]), return_sequences=target_rep, stateful=stateful))
        else:
            model.add(Masking(mask_value=0., input_shape=input_shape))
            model.add(LSTM(256, return_sequences=target_rep, stateful=stateful))
    #model.add(BatchNormalization())
    model.add(Activation('tanh'))
    #model.add(Dropout(0.5))

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
        model.add(Dense(1, activation = 'sigmoid'))
    #model.add(BatchNormalization())
    optimizer = Adam(lr=1e-4, beta_1 =.5 )
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return (model)

def hierarchal_lstm (input_shape, embedding, stateful= False, target_rep = False, bidirectional = False):
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
    tr_auc = data[0]['tr_auc']
    tr_auc = [i for i in tr_auc]
    tr_loss = data[0]['tr_loss']
    tr_loss = [i[0] for i in tr_loss]
    te_auc = data[0]['te_auc']
    te_auc = [i for i in te_auc]

    %matplotlib qt
    plt.plot(tr_loss)
    plt.plot(tr_auc)
    plt.plot(te_auc)
    plt.title(title)
    plt.ylabel('performance')
    plt.xlabel('epoch')
    plt.legend(['tr_loss', 'tr_auc', 'te_auc'], loc='best')
    plt.show()
    
    dat = {}
    te_matrix = []; te_auc = []
    for idx in range(1, len(data)+1):
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
    df = pd.DataFrame(dat)
    df = df.transpose()
    df.index.name = 'KFOLD'
    df.to_pickle(pkl)
    #df.read_pickle(pkl) to test.    
    
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

