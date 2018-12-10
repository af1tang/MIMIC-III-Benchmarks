#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:25:50 2018

@author: af1tang
"""
import string
import nltk
import pickle
import pandas as pd
import random
import numpy as np
import progressbar
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb

from utilities import *

### Change the Path to UMLS codes files ###
path_parsed = '/home/af1tang/Desktop/mimic3-parsed/'    #parsed MIMIC III disch. notes.
path_icd = 'all_w2v_CNL.txt'                            #parsed CNL codes.
path_words = 'all_w2v_UMLS.txt'                         #parsed UMLS codes.


def process_notes(keys, table):
    '''keys: list of UMLS codes
    table: {subject_id: {hadm_id: 24 timesteps}} 
    output: {hadm_id: [code1, code2, ... ]}'''
    import glob
    import re
    from datetime import datetime, timedelta 
    
    filename = path_parsed
    files = glob.glob(filename + '/*.text')
    subj = sorted([int(f.split('-')[-2]) for f in files])
    dct = {}
    
    subj = [s for s in subj if s in table.keys()]
    
    for idx in progressbar.progressbar(range(len(subj))):
        s = subj[idx]
        with open(path_parsed + 'notes-'+ str(s) + '-dx.text', 'r') as f:
            text = f.read().lower()
        text = text.split('## ')
                    
        dsch = [(datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', tx).group(), '%Y-%m-%d').date(), tx.split(' ')[2])
                    for tx in text if "discharge summary" in tx]

        if len(dsch) >0:
            dsch = int( dsch[np.argmax([d[0] for d in dsch])][1] )
            notes = []
            for i in text[1:]:
                ## match for time: if no time of note, then discard ##
                match = re.search(r'\d{4}-\d{2}-\d{2}', i)
                if match:
                    date = datetime.strptime(match.group(), '%Y-%m-%d').date()
                    time = re.search(r'\d{2}:\d{2}:\d{2}', i)
                    if time:
                        time = datetime.strptime(time.group(), '%H:%M:%S').time()
                        time = pd.to_datetime(datetime.combine(date, time))
                    else: 
                        time = pd.to_datetime(date)
                    h = [ k for k,v in list(table[s].items()) if v == find_nearest(list(table[s].values()), time)][0]
                    words = []
                    for term in i.split(' '):
                        if term in keys:
                            word = term
                        elif term[1:] in keys:
                            word = term[1:]
                        elif term[0:-1] in keys:
                            word = term[0:-1]
                        else:
                            word = None
                        if word:
                            words.append(word)
                    if (len(words) > 0) and (abs(table[s][h] -time) < timedelta(days=1)) and ("discharge" not in i):
                        notes.append( (words, h) )
            dct.update(dict([(h, list(set(flatten([w for w,n in notes if n == h]))) ) for h in table[s].keys()]) )
    return dct

def disch_notes(keys):
    '''keys: list of UMLS codes'''
    import glob
    import re
    import datetime
    filename = path_parsed
    files = glob.glob(filename + '/*.text')
    subj = sorted([int(f.split('-')[-2]) for f in files])
    dct = {}
    
    for j in progressbar.progressbar(range(len(subj))):
        s = subj[j]
        with open(filename + 'notes-'+ str(s) + '-dx.text', 'r') as f:
            text = f.read().lower()
        text = text.split('## ')
        dsch = [x for x in text if "discharge summary" in x]
        notes = []
        for i in dsch:
            words = []
            for term in i.split(' '):
                if term in keys:
                    word = term
                elif term[1:] in keys:
                    word = term[1:]
                elif term[0:-1] in keys:
                    word = term[0:-1]
                else:
                    word = None
                if word:
                    words.append(word)
            if len(words) > 0:
                match = re.search(r'\d{4}-\d{2}-\d{2}', i)
                date = datetime.datetime.strptime(match.group(), '%Y-%m-%d').date()
                date = pd.to_datetime(date)
                notes.append( (words, pd.to_datetime(date)) )
        dct[s] = notes
    return dct

def get_icd_dict():
    with open(path_parsed + path_icd, 'r') as f:
        text = f.read().lower()
    text = text.split('\n')
    text = [i for i in text if i[0:3] == 'idx']
    items = [i.split(' ') for i in text] 
    items = [(i[0], np.array([float(k) for k in i[1:] if len(k) > 0]) ) for i in items]
    dct = dict(items)
    return dct

def get_umls_dict():
    with open(path_parsed + path_words, 'r') as f:
        text = f.read().lower()
    text = text.split('\n')
    text = [i for i in [i for i in text if len(i) > 0] if i[0] == 'c']
    items = [i.split(' ') for i in text] 
    items = [(i[0], np.array([float(k) for k in i[1:] if len(k) > 0]) ) for i in items]
    dct = dict(items)
    return dct
