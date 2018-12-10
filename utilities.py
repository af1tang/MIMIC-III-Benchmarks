import progressbar
import pandas as pd
import numpy as np


def one_hot(arr, size):
    onehot = np.zeros((len(arr),size), dtype = int)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):            
            onehot[i, int(arr[i])]=1
    #onehot[np.arange(len(arr)), arr] =1
    return onehot

def bow_sampler(x, size):
    if not pd.isnull(x).all():
        bow = np.sum(one_hot(x, size), axis=0) 
        bow = np.array([(lambda x: 1 if x >0 else 0)(xx) for xx in bow])
        first = one_hot(x,size)[0]
        last = one_hot(x,size)[-1]
        return [first, bow, last]
    else:
        return np.nan

def find_prev(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == 0:
        return array[idx]
    else:
        return array[idx-1]

def find_next(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == len(array) -1:
        return array[idx]
    else:
        return array[idx+1]
    
def flatten(lst):
    make_flat = lambda l: [item for sublist in l for item in sublist]
    return make_flat(lst)


def idx_2_OHV (idx, size):
    tmp = [0]*size
    try:
        tmp[idx] = 1
    except: print(idx)
    return np.array(tmp)

def flatten(lst):
    make_flat = lambda l: [item for sublist in l for item in sublist]
    return make_flat(lst)    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]