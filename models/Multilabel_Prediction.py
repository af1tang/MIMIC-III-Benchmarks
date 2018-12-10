# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 14:45:08 2017

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
    maxlen = 48
    for idx in range(len(X)):
        if len(X[idx]) > maxlen:
            #X[idx] = np.concatenate((X[idx][0:24], X[idx][-24:]), axis = 0)
            X[idx] = X[idx][-48:]
    X= sequence.pad_sequences (X, maxlen, dtype = 'float32')

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
    for train_index, test_index in skf.split(X, Y[:, -1]):
        count +=1
        print ("KFold #{0}, TIME: {1} Hours".format(count, (time.time() - start)/3600))
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    
        
        #if time distributed, must add new axes to Y's
        if (target_rep == True):
            y_train = np.repeat(y_train[:, np.newaxis, :], maxlen, axis=1)
        
        model = lstm_model(input_shape = (X_train.shape[1], X_train.shape[2]), stateful = stateful, target_rep = target_rep, bidirectional = bidirectional)
        
        data[count] = {}
        data[count]['tr_auc'] = []
        data[count]['f1_score'] = []
        data[count]['tr_loss'] = []
        data[count]['te_auc'] = []
        data[count]['te_matrix'] = []
        
        for epoch in range(30):
            #start = time.time()
            #fix class imbalance w/ class weights
            #sample_wt = class_weight.compute_sample_weight('balanced', y_train)
            
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
            
            
            fpr = dict()
            tpr = dict()
            tr_roc_auc = dict()
            f1= dict()
            te_roc_auc = dict()
            te_matrix = dict()
            
            if (target_rep == True):
                y_pred = y_pred[:, -1, :]
                y_train_last = y_train[:, -1, :]
                yhat = yhat[:, -1, :]                       
                for idx in range(Y[0].shape[0]):
                    fpr[idx], tpr[idx], _ = roc_curve(y_train_last[:, idx], y_pred[:, idx])
                    tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                    
                # Compute micro-average ROC curve and ROC area 
                fpr["micro"], tpr["micro"], _ = roc_curve(y_train_last.ravel(), y_pred.ravel())
                tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            else:
                for idx in range(Y[0].shape[0]):
                    fpr[idx], tpr[idx], _ = roc_curve(y_train[:, idx], y_pred[:, idx])
                    tr_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                    
                # Compute micro-average ROC curve and ROC area 
                fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_pred.ravel())                
                tr_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            #print progress            
            print('EPOCH {0}'.format(epoch))
            print('AUROC (top dx): {0}'.format(tr_roc_auc[0]))
            print('AUROC (2nd dx): {0}'.format(tr_roc_auc[1]))
            print('AUROC (3rd): {0}'.format(tr_roc_auc[2]))
            print('AUROC (4th): {0}'.format(tr_roc_auc[3]))
            print('AUROC (Micro) =  {0}'.format(tr_roc_auc['micro']))
            print('Time: {0}'.format((time.time() - start)/60))
            print ('++++++++++++++++++')
            
            data[count]['tr_auc'].append(tr_roc_auc)
            data[count]['tr_loss'].append(history.history['loss'])
            
            for idx in range(Y[0].shape[0]):
                fpr[idx], tpr[idx], _ = roc_curve(y_test[:, idx], yhat[:, idx])
                te_roc_auc[idx] = auc(fpr[idx], tpr[idx])
                te_matrix[idx] = confusion_matrix(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                f1[idx] = f1_score(y_test[:, idx], np.array([round(i) for i in yhat[:, idx]]))
                
            # Compute micro-average ROC curve and ROC area 
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yhat.ravel())
            te_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            f1["micro"] = f1_score(y_test.ravel(), np.array([round(i) for i in yhat.ravel()]))
            #te_matrix['micro'] = confusion_matrix(y_test.ravel(), np.array([round(i) for i in yhat.ravel()]))

            data[count]['f1_score'].append(f1)
            data[count]['te_matrix'].append(te_matrix)
            data[count]['te_auc'].append(te_roc_auc)
        
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
    model.add(Dropout(0.5))

    #if (bidirectional == True):
    #    model.add(Bidirectional(LSTM(512, return_sequences = target_rep, stateful = stateful), merge_mode = 'concat'))
    #else:
    #    model.add(LSTM(512, return_sequences = target_rep, stateful = stateful)) 
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.5))
    
    if (target_rep == True):
        model.add(TimeDistributed(Dense(25, activation = 'sigmoid')))
    else:
        model.add(Dense(25, activation = 'sigmoid'))
    #model.add(BatchNormalization())
    optimizer = Adam(lr=1e-4, beta_1 = .5)
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
    y = Dense(4, activation = 'sigmoid') (xx)
    model = Model(input = [x, dx], output = [y])
    model.compile (loss = 'binary_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return (model)
    
def plotting(loss, title, data, pkl):
    #title = "Vanilla BILSTM on Multilabel Predictions"
    tr_auc = data[0]['tr_auc']
    re_tr_auc = [i[0] for i in tr_auc]
    morth_tr_auc = [i[1] for i in tr_auc]
    mort30_tr_auc = [i[2] for i in tr_auc]
    mort1y_tr_auc = [i[3] for i in tr_auc]
    micro_tr_auc = [i['micro'] for i in tr_auc]
    tr_loss = data[0]['tr_loss']
    tr_loss = [i[0] for i in tr_loss]
    tr_acc = data[0]['tr_acc']
    tr_acc = [i[0] for i in tr_acc]
    
    te_auc = data[0]['te_auc']
    re_te_auc = [i[0] for i in te_auc]
    morth_te_auc = [i[1] for i in te_auc]
    mort30_te_auc = [i[2] for i in te_auc]
    mort1y_te_auc = [i[3] for i in te_auc]
    micro_te_auc = [i['micro'] for i in te_auc]
    
    te_acc = data[0]['te_acc']
    re_te_acc = [i[0] for i in te_acc]
    morth_te_acc = [i[1] for i in te_acc]
    mort30_te_acc = [i[2] for i in te_acc]
    mort1y_te_acc = [i[3] for i in te_acc]
    micro_te_acc = [i['micro'] for i in te_acc]
    
    %matplotlib qt
    plt.plot(tr_loss)
    plt.plot(tr_acc)
    plt.plot(re_tr_auc)
    plt.plot(re_te_auc)
    plt.plot(re_te_acc)
    plt.plot(morth_tr_auc)
    plt.plot(morth_te_auc)
    plt.plot(morth_te_acc)
    plt.plot(mort30_tr_auc)
    plt.plot(mort30_te_auc)
    plt.plot(mort30_te_acc)
    plt.plot(mort1y_tr_auc)
    plt.plot(mort1y_te_auc)
    plt.plot(mort1y_te_acc)
    plt.plot(micro_tr_auc)
    plt.plot(micro_te_auc)
    plt.plot(micro_te_acc)
    
    plt.title(title)
    plt.ylabel('performance')
    plt.xlabel('epoch')
    plt.legend(['tr_loss', 'tr_acc', 're tr_auc', 're te_auc', 're te_acc', 
    'mort_h tr_auc', 'mort_h te_auc', 'mort_h te_acc', 
    'mort_30 tr_auc', 'mort_30 te_auc', 'mort_30 te_acc',
    'mort_1y tr_acc', 'mort_1y te_auc', 'mort_1y te_acc',
    'micro tr_auc', 'micro te_auc', 'micro te_acc'])
    plt.show()
    
    #saving dataframe
    dat = {}
    dat['tr_loss'] = tr_loss
    dat['tr_acc'] = tr_acc
    dat['re_tr_auc'] = re_tr_auc
    dat['re_te_auc'] = re_te_auc
    dat['re_te_acc'] = re_te_acc
    dat['mort_h tr_auc'] = morth_tr_auc
    dat['mort_h te_auc'] = morth_te_auc
    dat['mort_h te_acc'] = morth_te_acc
    dat['mort_30 tr_auc'] = mort30_tr_auc
    dat['mort_30 te_auc'] = mort30_te_auc
    dat['mort_30 te_acc'] = mort30_te_acc
    dat['mort_1y tr_auc'] = mort1y_tr_auc
    dat['mort_1y te_auc'] = mort1y_te_auc
    dat['mort_1y te_acc'] = mort1y_te_acc
    df = pd.DataFrame(dat)
    df.to_pickle('/home/andy/Desktop/MIMIC/dat/multilabelvanilla_bilstm.pkl')

if __name__ == '__main__':
    X = np.load(x_file)
    Y = np.load(y_file)
    dx = np.load(sentence_file)
    main(X, Y, dx)