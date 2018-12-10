#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 00:00:52 2018

@author: af1tang
"""
import pickle
import pandas as pd
import numpy as np
import time
import progressbar
from datetime import datetime, timedelta

## Set Paths ##
path_tables = '/home/af1tang/Desktop/local_mimic/tables/'
path_views = '/home/af1tang/Desktop/local_mimic/views/'
path_notes = '/home/af1tang/Desktop/local_mimic/notes/'
path_vars = '/home/af1tang/Desktop/vars/'
path_views = '/Users/af1tang/Dropbox/MIT/vars/pivot_time'
###############

#####################
### PREPROCESSING ###
#####################
def discharge_notes(dsch, times):
    '''pre: dsch = {subject_id: ([code], ttime)}, times = {subject_id: {hadm_id:time}}
    post: dct = {hadm_id: code}
    '''
    dct = {}
    for s in times.keys():
        #hadm = dict([(v,k) for k,v in  times[s].items()])
        dsch_t = dict([(t,c) for (c,t) in dsch[s]])
        for h in times[s].keys():
            c = dsch_t[find_nearest(list(dsch_t.keys()), times[s][h])]
        #for c, t in dsch[s]:
         #   h = hadm[find_nearest(list(hadm.keys()), t)]
            if (abs(find_nearest(list(dsch_t.keys()), times[s][h]) -times[s][h]) < timedelta(days=30)):
                dct[h]= c
    return dct

def filtered_icd9(dct_freq, dct):
    '''pre: dct_freq = frequency of icd9s, dct = {subject_id: {hadm_id: icd9 list}}
    post: features = {hadm_id: filtered icd9s}, freq = [(icd9, descr, count),...]
    criteria: only consider icd9 appearing in 1%+ of admissions. 
    '''
    dct2 = {}
    for s in dct.keys():
        for h in dct[s].keys():
            dct2[h] = dct[s][h]
    dct = dct2; del dct2
    
    freq = [(k,d,v) for (k,d,v) in dct_freq if v >= .01 *len(dct)]
    lst = [k for (k,d,v) in freq]
    print (lst[-1])
    features = {}
    for h in dct.keys():
        notes = [note for note in dct[h] if note in lst]
        if len(notes) > 0:
            features[h] = notes
    return features, freq

def composite(dsch, memories, labels, Xy):
    '''pre: Xy = {hadm_id: (x_t, y)}, labels = {hadm_id: y}, 
    dsch = {hadm_id: c}, memories = {h: x}
    post: features = {hadm_id: (x_t, x, c, y)}
    '''
    features = {}
    lst = sorted(set(Xy.keys()).intersection(set(dsch.keys())).intersection(labels.keys()))
    for i in progressbar.progressbar(range(len(lst))):
        h = lst[i]
        features[h] = ( Xy[h], memories[h], np.array(dsch[h]), np.array(labels[h]) )
    return features

def build_memory(dct, encoder, timesteps = 24, n_features=195):
    '''dct: {hadm_id: (X, y)}
    encoder: attn encoder, compresses temporal series into vector
    '''
    memories = {}
    lst = sorted(dct.keys())
    exceptions = []
    for i in progressbar.progressbar(range(len(lst))):
        h = lst[i]
        seq = dct[h][0]
        length = seq.shape[0]
        if length >=6:
            offset = timesteps - seq.shape[0]
            mat = np.zeros((timesteps,n_features))
            mat[offset:seq.shape[0] + offset, :seq.shape[1]] = seq
            #seq = np.array(list(window(mat, n=6)))
            seq = np.array(np.split(mat, timesteps // 6))
            seq = np.array([s for s in seq if 0 not in [lg.norm(ss - np.zeros((1,n_features))) for ss in s ]])
            #seq = np.array([s for s in seq if np.sum(s)>0])
        else:
            offset = 6 - seq.shape[0]
            mat = np.zeros((6,n_features))
            mat[offset:seq.shape[0] + offset, :seq.shape[1]] = seq
            seq = mat.reshape(1, 6, n_features)
        try:
            memories[h] = encoder.predict(seq)
        except: exceptions.append(h); print(h, len(seq))
    return memories, exceptions

####################
### PIVOT TABLES ###
####################
def pivot_icd(table, w2v_icd, file_names = '/home/af1tang/Desktop/local_mimic/tables/d_icd_diagnoses.csv'):
    '''table: {s: {h: t_24}}
    w2v_icd: {idx: w2v} '''
    df = pd.read_csv('/home/af1tang/Desktop/local_mimic/tables/diagnoses_icd.csv')
    #process w2v
    w2v_icd = dict([(''.join(k[4:].split('.')), v) for k, v in w2v_icd.items()])
    #icd names
    icd_names = pd.read_csv('/home/af1tang/Desktop/local_mimic/tables/d_icd_diagnoses.csv')
    #make dictionary of icd9 codes
    dct = {}
    subj = sorted(table.keys())
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        dictionary = df[(df.subject_id == s)][['hadm_id', 'icd9_code']].groupby('hadm_id')['icd9_code'].apply(list).to_dict()
        dictionary = dict([(k,v ) for k,v in dictionary.items() if k in table[s].keys()])
        dct[s] = dictionary
    lengths = [dct[i].values() for i in dct.keys()]
    lengths = flatten(lengths)
    lengths = flatten(lengths)
    unique, counts = np.unique(lengths, return_counts=True)
    #frequency dictionary
    dct_freq = dict(zip(unique, counts))
    items = sorted(dct_freq.items(), key = lambda x: x[1], reverse = True)
    items = [(k,v) for k,v in items if k in w2v_icd]
    ## add names ##
    common = list(set(icd_names.icd9_code).intersection([i[0] for i in items]))
    common = icd_names[icd_names.icd9_code.isin(common)]
    common = common[['icd9_code', 'short_title']].groupby('icd9_code')['short_title'].apply(list).to_dict()
    dct_freq = []
    for idx, count in items:
        if idx in common.keys():
            dct_freq.append((idx, common[idx][0], count))
    return dct, dct_freq

def pivot_std_time(dct):
    p_bg = pd.read_pickle(path_views + '/pivot_bg')
    p_gcs = pd.read_pickle(path_views + '/pivot_gcs')
    #p_gcs = p_gcs[['icustay_id', 'charttime', 'gcs']]
    p_uo = pd.read_pickle(path_views+'/pivot_uo')
    p_vital= pd.read_pickle(path_views + '/pivot_vital')
    p_lab = pd.read_pickle(path_views + '/pivot_lab')
    cohort = pd.read_csv(path_views + '/icustay_detail.csv')
    ## Exclusion criteria ##
    cohort = cohort[(cohort.age>=18)&(cohort.los_hospital>=2)&(cohort.los_icu>1) &
                    (cohort.subject_id.isin(dct.keys()))]
    ## hourly binning ##
    p_bg.charttime = pd.to_datetime(p_bg.charttime)
    p_bg = p_bg.dropna(subset=['hadm_id'])
    p_vital.charttime = pd.to_datetime(p_vital.charttime)
    p_vital = p_vital.dropna(subset=['icustay_id'])
    p_lab.charttime = pd.to_datetime(p_lab.charttime)
    p_lab = p_lab.dropna(subset=['hadm_id'])
    p_uo.charttime = pd.to_datetime(p_uo.charttime)
    p_uo = p_uo.dropna(subset=['icustay_id'])
    p_gcs.charttime = pd.to_datetime(p_gcs.charttime)
    p_gcs = p_gcs.dropna(subset=['icustay_id'])
    
    ## initialize icustays dict ##
    dct_bins = {}
    lst= sorted(list(set(cohort.hadm_id)))
    hadm_dct = dict([(h, cohort[cohort['hadm_id']==h].subject_id.values[0]) for h in lst])
    
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    dfs= [p_bg, p_lab, p_vital, p_uo, p_gcs]
    cols = [['pco2', 'fio2_chartevents', 
       'aado2_calc', 'pao2fio2ratio', 'ph', 'baseexcess',
       'totalco2',  'methemoglobin',  'calcium',  'tidalvolume', 'peep'],        
        ['aniongap', 'albumin', 'bilirubin', 'creatinine', 'platelet', 'bicarbonate',
         'hematocrit', 'hemoglobin', 'potassium', 'sodium', 'lactate',
       'ptt', 'inr', 'pt', 'bun', 'wbc'],
         ['heartrate', 'sysbp', 'diasbp', 'meanbp',
       'resprate', 'tempc', 'spo2', 'glucose'],
        ['urineoutput'],
        ['gcs', 'gcseyes', 'gcsmotor', 'endotrachflag']]
    
    ## initialize features by filtered hadm ##
    features = {}
    subj = sorted(set(cohort.subject_id))
    exceptions = []
    for i in progressbar.progressbar(range(len(subj))):
        s = subj[i]
        hadm = np.array([h for h in set(cohort[cohort.subject_id == s].hadm_id) if pd.to_datetime(datetime.strptime(cohort[cohort.hadm_id==h].admittime.values[0], 
                                          '%Y-%m-%d %H:%M:%S')) < dct[s][np.argmax([idx[1] for idx in dct[s]])][1] + timedelta(hours=24)])
        if len(hadm>0):
            hadm = hadm[np.argmax([np.sum([len(p_vital[p_vital.icustay_id == icu_hadm[h][ii]]) 
                                            for ii in range(len(icu_hadm[h]))]) + 
                                     len(p_lab[p_lab.hadm_id==h]) for h in hadm])]
            timesteps = [pd.to_datetime(datetime.strptime(cohort[cohort.hadm_id==hadm].admittime.values[0], 
                                              '%Y-%m-%d %H:%M:%S') + timedelta(hours=hr)) for hr in range(24)]
            timesteps = [tt.replace(microsecond=0,second=0,minute=0) for tt in timesteps]
            features[hadm] = {}
            for t in timesteps:
                features[hadm][t] = {}
        else:
            exceptions.append(s)

    ## eliminate low time-step samples ##
    lst = []
    #initialize timestamps with vital signs
    for j in progressbar.progressbar(range(len(icustays))):
        h = icustays[j]
        if icu_dct[h] in features.keys():
            timesteps = [i for i in p_vital[p_vital['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[icu_dct[h]].keys())]
            if len(timesteps) >= 6:
                lst.append(icu_dct[h])

    #get timestamps for labs
    lst2 = []
    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_lab[p_lab['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist() if i <= max(features[h].keys())]
        if len(timesteps)>=1:
            lst2.append(h)
    lst = lst2; del lst2
    #update icustays list and features
    features = dict([(k,v) for k,v in features.items() if k in lst])
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())

    #from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    #gpr = GPR()
    
    lsts = [lst, lst, icustays, icustays, icustays]
    for idx in range(len(dfs)):
        #x = dfs[idx].drop('charttime',1)
        #try:
        #    x = x.drop('icustay_id',1)
        #except:
        #    x = x.drop('hadm_id',1)
        #dfs[idx] = dfs[idx][((x > x.quantile(.05)) & (x < x.quantile(.95))).all(1)]
        for c in cols[idx]:
            dfs[idx][c] = (dfs[idx][c]-dfs[idx][c].min() )/ (dfs[idx][c].quantile(.95) - dfs[idx][c].min())
            #dfs[idx][c][dfs[idx][c] >=1] = 1
            print(c)
            #for each admission, for each hourly bin, compile bow vector for features
            for i in progressbar.progressbar(range(len(lsts[idx]))):
                h = lsts[idx][i]
                if len(lst) == len(lsts[idx]):
                    s = dfs[idx][dfs[idx]['hadm_id']==h].set_index('charttime')[c]
                else:
                    s =  dfs[idx][dfs[idx]['icustay_id']==h].set_index('charttime')[c]
                    h = icu_dct[h]
                
                try:
                    s = s.interpolate('from_derivatives', 
                                      limit_direction = 'both', limit_area = 'inside')
                except:
                    s = s.interpolate(limit_direction = 'both', limit_area = 'inside')
                s = s.fillna(0.0)
                #s[s >= 5] = 5
                #s[s <=-5] = 5
                time_range= sorted(features[h].keys())
                s = s.loc[time_range[0]: time_range[-1]]
                if len(s)>0:
                    #s = s.interpolate('time', limit_direction='both')
                    try:
                        s = s.resample('H').ohlc()['close'].interpolate('from_derivatives', limit_direction = 'both')
                    except:
                        s= s.resample('H').ohlc()['close'].interpolate(limit_direction='both')
                    #xx = (s.index - s.index.min()) / np.timedelta64(1, 'h')
                    #yy = s.values
                    #gpr.fit([xx], [yy])
                    s = s.reindex(pd.to_datetime(time_range))
                    s = s.interpolate()
                    s = s.fillna(0.0)
                    s[s>=1] = 1.0
                    s[s<=0] = 0.0
                    s = dict([(key,val) for key,val in s.items() if key <= max(features[h].keys())])
                    times = sorted(s.keys())
                    for t in time_range:
                        if t < times[0]:
                            features[h][t][c] = s[times[0]]
                        elif t in times:
                            features[h][t][c] = s[t]
                        elif t not in s.keys():
                            curr = find_nearest(times, t)
                            features[h][t][c] = s[curr]
                            s[t] = s[curr]
                        else:
                            print(times, t)
                    if pd.isnull(list(s.values())).any():
                        print(s)
                else:
                    for t in sorted(features[h].keys()):
                        features[h][t][c] = 0.0

    return features, dct_bins

