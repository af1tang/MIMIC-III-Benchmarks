#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:45:26 2018

@author: af1tang
"""

import pickle
import pandas as pd
import numpy as np
import time
import progressbar

from utilities import *

## Set Paths ##
path_tables = '/home/af1tang/Desktop/local_mimic/tables/'
path_views = '/home/af1tang/Desktop/local_mimic/views/'
path_notes = '/home/af1tang/Desktop/local_mimic/notes/'
path_vars = '/home/af1tang/Desktop/vars/'
###############

#############################
#### COHORT ACQUISITION #####
############################
def make_labels():
    cohort = pd.read_csv(path_views + 'cohorts/extub_cohorts.csv')
    icu_details = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    vent = pd.read_csv(path_views + 'vent/ventdurations.csv')
    
    extub, vent_multi, vent_subj = extub_cohort()
    #exclusion filter
    excluded = list(set(cohort[cohort.excluded==1].subject_id))
    filtered = [i for i in list(set(cohort.subject_id)) if i not in excluded]
    subj = list(set(filtered).intersection(set(vent_subj)))    #N = 21,101 
    ext_subj = list(set([i[0] for i in extub]))
    
    #make labels
    dct = {}
    for s in subj:
        if s in ext_subj:
            lst = [i for i in extub if i[0] == s]
            lst = sorted(lst, key=lambda x: icu_details[icu_details.hadm_id==x[1]].admittime.values[0])[-1]
            stays = sorted(lst[-1], key = lambda x: pd.to_datetime(x[1]))
            dct[s] = {'hadm_id': lst[1], 'icustay_id': stays[-1][0], 'mv_onset': stays[-1][1], 
                       'mv_extub': stays[-1][2], 'reint': stays[-1][3], 'extub_fail': 1, 
                       'los_hospital': icu_details[(icu_details.hadm_id == lst[1]) & (icu_details.icustay_id == stays[-1][0])].los_hospital.values[0],
                       'los_icu': icu_details[(icu_details.hadm_id == lst[1]) & (icu_details.icustay_id == stays[-1][0])].los_icu.values[0],
                       'mort30': cohort[(cohort.icustay_id == stays[-1][0])&(cohort.hadm_id == lst[1])].thirtyday_expire_flag.values[0],
                       'morth': icu_details[(icu_details.hadm_id == lst[1]) & (icu_details.icustay_id == stays[-1][0])].hospital_expire_flag.values[0]}
        else:
            select = pd.merge(left = icu_details[icu_details.subject_id == s][['hadm_id', 'admittime', 'icustay_id', 'los_icu', 'los_hospital', 'hospital_expire_flag']].sort_values(by=['admittime']), right = vent, on = 'icustay_id').head(1)
            hadm, icu_id, los_icu, los_hosp, morth = select.hadm_id.values[0], select.icustay_id.values[0], select.los_icu.values[0], select.los_hospital.values[0], select.hospital_expire_flag.values[0]
            mv_onset, mv_off = vent[vent.icustay_id ==icu_id].starttime.values[0], vent[vent.icustay_id ==icu_id].endtime.values[0]
            dct[s] = {'hadm_id': hadm, 'icustay_id': icu_id, 'mv_onset': mv_onset, 'mv_extub': mv_off, 'reint': None,
                       'extub_fail':0, 'los_hospital': los_hosp, 'los_icu': los_icu, 
                       'mort30': cohort[(cohort.icustay_id == icu_id)&(cohort.hadm_id == hadm)].thirtyday_expire_flag.values[0],
                       'morth': morth}
    return dct

def make_vali(dct):
    icu_details = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    dx_df = pd.read_csv('/home/af1tang/Desktop/local_mimic/tables/diagnoses_icd.csv')
    icds = {'51882':'ARDS', '99731': 'VAP', '4281': 'pulm_edema', '4160': 'pulm_htn', '4168': 'pulm_htn',
            '7994': 'cachexia', '45340': 'DVT', '5184': 'lung_injury', '51881': 'resp_fail',
            '7704': 'atelecstasis', '27669': 'fluid_overload'}
    icd_idx = {'51882':0, '99731': 1, '4281': 2, '4160': 3, '4168': 3,
        '7994': 4, '45340': 5, '5184': 6, '51881': 7,
        '7704': 8, '27669': 9}
    
    #arrange icu_details by admittime of hospital admission
    icu_times = icu_details[['subject_id', 'hadm_id', 'admittime']].sort_values(by=['subject_id', 'admittime'])
    
    #build diagnostic history table
    df = pd.merge(left = icu_times, right = dx_df[['hadm_id', 'icd9_code']], how='inner', on = 'hadm_id')
    df = df.drop_duplicates()
    
    #add VALI to dct
    count = 0
    for s in dct.keys():
        #select = df[(df.subject_id == s) & (dct[s]['mv_onset'] <= pd.to_datetime(df.admittime)) ]
        select = df[(df.subject_id == s) & (dct[s]['hadm_id'] == df.hadm_id)]
        if len(select) >0:
            vec = [0]*10
            lst = list(set(select.icd9_code))
            for item in lst:
                if item in icd_idx.keys():
                    vec[icd_idx[item]] = 1
            dct[s]['VALI']= vec
        else:
            dct[s]['VALI'] = None
        count+=1 
        print(count, end=' ')
    return dct, icds, icd_idx, df

def extub_cohort(path = path_views+'vent/ventdurations.csv' ):
    '''pre: path to vent duration view
        post: extub: (s, h, [[icustay_id, starttime, endtime, reintubtime], ...]) of intubations
    '''
    vent = pd.read_csv(path)
    icustays = pd.read_csv(path_tables + 'icustays.csv')
    df = pd.merge(left = icustays, right = vent, how = 'inner', on = 'icustay_id')
    df = df[['subject_id', 'hadm_id','icustay_id', 'starttime', 'endtime','duration_hours', 'ventnum']]
    subj = list(set(df.subject_id))
    multi = list(set(df[df.ventnum>1].subject_id))          #4500 
    singles = [i for i in subj if i not in multi]           #18,757 patients
    print("multi: {0}, singles:{1}".format(len(multi), len(singles)))
    
    extub= []  
    for s in multi:
        hadm =list(set(df[df.subject_id == s].hadm_id))
        for h in hadm:
            ext_fail = []
            add_on = False
            starts = list(set(df[(df.subject_id == s) & (df.hadm_id == h)].starttime))
            ends = list(set(df[(df.subject_id == s) & (df.hadm_id == h)].endtime)) 
            starts = sorted( [pd.to_datetime(i) for i in starts] )
            ends = sorted( [pd.to_datetime(i) for i in ends] )
            for ii in range(len(starts)-1):
                if (pd.to_datetime(starts[ii+1]) - pd.to_datetime(ends[ii]))/np.timedelta64(1, 'h') < 48.0:
                    add_on= True
                    icu_id = df[(df.hadm_id == h) & (pd.to_datetime(df.starttime) == starts[ii])].icustay_id.values[0]
                    ext_fail.append([icu_id, pd.to_datetime(starts[ii]), pd.to_datetime(ends[ii]), pd.to_datetime(starts[ii+1])])
            if add_on == True:
                extub.append((s, h, ext_fail))        
    return extub, multi, subj

def exclude_trach():
    #trach patients
    #exclude from final cohort
    df = pd.read_pickle(path_vars + '/pivot_dx/pivot_icd')
    df = df[df.V440 == 1]; lst = list(set(df.index))
    #gcs
    gcs = pd.read_pickle(path_vars + 'pivot_time/pivot_gcs')
    #original cohort
    cohort = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    cohort = cohort[['subject_id', 'hadm_id', 'icustay_id','intime', 'outtime', 'admittime']]
    low_gcs = gcs[gcs['gcs']<8.0][['icustay_id', 'gcs']].dropna()
    low_gcs = pd.merge(cohort, low_gcs, left_on = 'icustay_id', right_on = 'icustay_id')
    #low gcs cohort
    gcs_lst = list(set(low_gcs.subject_id))
    
    with open(path_vars + 'labels_dct', 'rb') as f:
        pre_dct = pickle.load(f)
    dct = {}
    for k in pre_dct.keys():
        if k not in lst:
            dct[k] = pre_dct[k] 
    return dct
        
#############################
#### FEATURE EXTRACTION #####
#############################
def pivot_dx():
    icu_details = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    icu_times = icu_details[['subject_id', 'hadm_id', 'admittime']].sort_values(by=['subject_id', 'admittime'])

    dx_df = pd.read_csv(path_tables + 'diagnoses_icd.csv')
    df = pd.merge(left = icu_times, right = dx_df[['hadm_id', 'icd9_code']], how='inner', on = 'hadm_id')
    df = df.drop_duplicates()
    for i in range(len(df)):
        df.iloc[i]['group'] = df.iloc[i]['icd9_code'][0:3]

    #unstack icd9 codes by 'get_dummies' -> groupby 'hadm_id' -> sum
    p = df[['subject_id', 'hadm_id', 'group']].set_index(['subject_id'])
    icd = pd.get_dummies(p, columns=['group'], prefix = '', prefix_sep='').groupby(['hadm_id'], as_index = True).sum()
    #icd = pd.get_dummies(p, columns=['icd9_code'], prefix = '', prefix_sep='', sparse=True).groupby(['hadm_id'], as_index = True).sum()
    #ICD-9 Pivot Table
    p = p.drop('group', axis = 1)
    p = p.drop_duplicates()
    icd = pd.merge(p, icd, left_on = 'hadm_id', right_index= True)
    
    #DRG Pivot Table
    drg = pd.read_csv(path_tables + 'drgcodes.csv')
    drg = pd.merge(left = icu_times, right = drg[['hadm_id', 'drg_code']], how = 'inner', on = 'hadm_id')
    drg = drg.set_index(['subject_id'])
    drg = drg.drop_duplicates()
    drgcodes = pd.get_dummies(drg, columns = ['drg_code'], prefix='', prefix_sep = '').groupby(['hadm_id'], as_index = True).sum()
    drg = drg.drop('drg_code', axis=1)
    drg = drg.drop_duplicates()
    drg = pd.merge(drg, drgcodes, left_on = 'hadm_id', right_index = True)
    #for ICD9 code, just change pivot to 'icd9_code' instead of 'group' 
    return icd, drg

def pivot_time():
    p_bg = pd.read_csv(path_views + '/pivots/pivoted_bg_art.csv')
    p_gcs = pd.read_csv(path_views + '/pivots/pivoted_gcs.csv')
    p_gcs = p_gcs[['icustay_id', 'charttime', 'gcs']]
    p_uo = pd.read_csv(path_views+'/pivots/pivoted_uo.csv')
    p_vital= pd.read_csv(path_views + '/pivots/pivoted_vital.csv')
    p_lab = pd.read_csv(path_views + '/pivots/pivoted_lab.csv')
    cohort = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    cohort = cohort[(cohort.age>=18)&(cohort.los_hospital>=1)&(cohort.los_icu>=1)]
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
    
    dct_bins = {}
    lst= sorted(list(set(cohort.hadm_id)))
    hadm_dct = dict([(h, cohort[cohort['hadm_id']==h].subject_id.values[0]) for h in lst])
    #initialize features
    features={}
    #for p_vital 
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    p_hadm = [p_bg, p_lab]
    p_icu = [p_uo, p_gcs]
    
    #initialize timestamps with vital signs
    for j in progressbar.progressbar(range(len(icustays))):
        h = icustays[j]
        timesteps = [i for i in p_vital[p_vital['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist()]
        timesteps = timesteps[0:48]
        if len(timesteps) > 18:
            if icu_dct[h] in features.keys():
                tmp = features[icu_dct[h]] 
                tmp += timesteps
                tmp = sorted(list(set(tmp)))
                tmp = tmp[0:48]
                features[icu_dct[h]] = tmp
            else:
                features[icu_dct[h]] = timesteps
    lst = sorted(features.keys())
    #get timestamps for UO and GCS
    for df in p_icu:
        for j in progressbar.progressbar(range(len(icustays))):
            h = icustays[j]
            timesteps = [i for i in df[df['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist()]
            timesteps = timesteps[0:48]
            if (icu_dct[h] in features.keys()) and (len(timesteps)>0):
                tmp = features[icu_dct[h]] 
                tmp += timesteps
                tmp = sorted(list(set(tmp)))
                tmp = tmp[0:48]
                features[icu_dct[h]] = tmp
    #get timestamps for labs
    features_new = {}
    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_lab[p_lab['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist()]
        timesteps = timesteps[0:48]
        if len(timesteps)>0:
            tmp = features[h]
            tmp += timesteps
            tmp = sorted(list(set(tmp)))
            features_new[h] =tmp
    features = features_new; del features_new; lst = sorted(list(set(features.keys())))

    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_bg[p_bg['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist()]
        timesteps = timesteps[0:48]
        if len(timesteps) > 0:
            tmp = features[h] 
            tmp += timesteps
            tmp = sorted(list(set(tmp)))
            features[h] = tmp
    #update icustays list
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    dfs= [p_bg, p_lab, p_vital, p_uo, p_gcs]
    lsts = [lst, lst, icustays, icustays, icustays]
    cols = [['so2', 'spo2', 'po2', 'pco2', 'fio2_chartevents', 
       'aado2_calc', 'pao2fio2ratio', 'ph', 'baseexcess', 'bicarbonate',
       'totalco2', 'hematocrit', 'hemoglobin', 'methemoglobin',  'calcium', 'temperature', 'potassium',
       'sodium', 'lactate', 'glucose',  'tidalvolume', 'peep'],        
        ['aniongap', 'albumin', 'bilirubin', 'creatinine', 'platelet',
       'ptt', 'inr', 'pt', 'bun', 'wbc'],
         ['heartrate', 'sysbp', 'diasbp', 'meanbp',
       'resprate', 'tempc', 'spo2', 'glucose'],
        ['urineoutput'],
        ['gcs']]
    #variables = (lambda l: [item for sublist in l for item in sublist])(cols)
    #add variables to each timestep
    for k in features.keys():
        tmp = {}
        for t in sorted(features[k])[0:48]:
            tmp[t] = {}
        features[k] = tmp

    for idx in range(len(dfs)):
        for c in cols[idx]:
            print(c)
            #get quintile bins
            bins = pd.qcut(dfs[idx][c], q=5, retbins = True, duplicates = 'drop')[1]
            dct_bins[c] = bins
            #for each admission, for each hourly bin, compile bow vector for features
            for i in progressbar.progressbar(range(len(lsts[idx]))):
                h = lsts[idx][i]
                if len(lst) == len(lsts[idx]):
                    s = dfs[idx][dfs[idx]['hadm_id']==h].set_index('charttime')[c]
                else:
                    s =  dfs[idx][dfs[idx]['icustay_id']==h].set_index('charttime')[c]
                    h = icu_dct[h]
                s = pd.cut(s, bins, labels = False)
                #s = s.resample('H').apply(lambda x: np.sum(one_hot(x,6), axis=0) if not pd.isnull(x).all() else np.array([0, 0, 0, 0, 0, 1])).to_dict()
                s = s.resample('H').apply(lambda x: bow_sampler(x, len(bins)-1)).to_dict()
                
                s = dict([(key,val) for key,val in s.items() if key in features[h].keys()])
                if pd.isnull(list(s.values())).all():
                    for t in sorted(features[h].keys()):
                        features[h][t][c] = np.zeros((len(bins)-1), dtype=int)
                else:
                    times = sorted([tt for (tt, val) in s.items() if not np.isnan(val).all()])
                    for t in sorted(features[h].keys()):
                        if t < times[0]:
                            features[h][t][c] = s[times[0]][0]
                        elif t in times:
                            features[h][t][c] = s[t][1]
                        elif t not in s.keys():
                            curr = find_nearest(sorted(s.keys()), t)
                            features[h][t][c] = s[curr][2]
                            s[t] = s[curr]
                        else:
                            prev = find_prev(sorted(s.keys()), t)
                            features[h][t][c] = s[prev][2]
                            s[t] = s[prev]

    return features, dct_bins
    
def pivot_labs(dct):
    p_bg = pd.read_csv(path_views + '/pivots/pivoted_bg_art.csv')
    p_gcs = pd.read_csv(path_views + '/pivots/pivoted_gcs.csv')
    p_gcs = p_gcs[['icustay_id', 'charttime', 'gcs']]
    p_uo = pd.read_csv(path_views+'/pivots/pivoted_uo.csv')
    p_vital= pd.read_csv(path_views + '/pivots/pivoted_vital.csv')
    p_lab = pd.read_csv(path_views + '/pivots/pivoted_lab.csv')
    cohort = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    #p_bg = pd.read_pickle('/Users/af1tang/Dropbox/MIT/vars/pivot_time/pivot_bg')
    #p_gcs = pd.read_pickle(path_views + '/pivot_gcs')
    #p_gcs = p_gcs[['icustay_id', 'charttime', 'gcs']]
    #p_uo = pd.read_pickle(path_views+'/pivot_uo')
    #p_vital= pd.read_pickle(path_views + '/pivot_vital')
    #p_lab = pd.read_pickle(path_views + '/pivot_lab')
    #cohort = pd.read_csv(path_views + '/icustay_detail.csv')
    #cohort = cohort[(cohort.age>=18)&(cohort.los_hospital>=1)&(cohort.los_icu>=1)]
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
    #df_bg = pd.DataFrame(index = p_bg.hadm_id.drop_duplicates())
    #df_bg = pd.DataFrame(index = p_bg['hadm_id'].drop_duplicates(), columns = p_bg.columns[2:])
    p_icu = [p_uo, p_gcs]
    
    dct_bins = {}
    hadm_dct = dict([(dct[k]['hadm_id'], k) for k in dct.keys()])
    #initialize features
    features = {}
    lst = sorted(hadm_dct.keys())
    #for p_vital 
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    #initialize timestamps with vital signs
    for j in progressbar.progressbar(range(len(icustays))):
        h = icustays[j]
        timesteps = [i for i in p_vital[p_vital['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist() if i<=dct[hadm_dct[icu_dct[h]]]['mv_extub']]
        timesteps = timesteps[-48:]
        if len(timesteps) > 6:
            if icu_dct[h] in features.keys():
                tmp = features[icu_dct[h]] 
                tmp += timesteps
                tmp = sorted(list(set(tmp)))
                tmp = tmp[-48:]
                features[icu_dct[h]] = tmp
            else:
                features[icu_dct[h]] = timesteps
    lst = sorted(features.keys())
    #get timestamps for UO and GCS
    for df in p_icu:
        for j in progressbar.progressbar(range(len(icustays))):
            h = icustays[j]
            timesteps = [i for i in df[df['icustay_id']==h].set_index('charttime').resample('H').first().index.tolist() if i<=dct[hadm_dct[icu_dct[h]]]['mv_extub']]
            timesteps = timesteps[-48:]
            if (icu_dct[h] in features.keys()) and (len(timesteps)>0):
                tmp = features[icu_dct[h]] 
                tmp += timesteps
                tmp = sorted(list(set(tmp)))
                tmp = tmp[-48:]
                features[icu_dct[h]] = tmp
    #get timestamps for labs
    features_new = {}
    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_lab[p_lab['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist() if i<=dct[hadm_dct[h]]['mv_extub']]
        timesteps = timesteps[-48:]
        if len(timesteps)>0:
            tmp = features[h]
            tmp += timesteps
            tmp = sorted(list(set(tmp)))
            tmp = tmp[-48:]
            features_new[h] =tmp
    features = features_new; del features_new; lst = sorted(list(set(features.keys())))

    for j in progressbar.progressbar(range(len(lst))):
        h = lst[j]
        timesteps = [i for i in p_bg[p_bg['hadm_id']==h].set_index('charttime').resample('H').first().index.tolist()]
        timesteps = timesteps[-48:]
        if len(timesteps) > 0:
            tmp = features[h] 
            tmp += timesteps
            tmp = sorted(list(set(tmp)))
            tmp = tmp[-48:]
            features[h] = tmp
    #update icustays list
    icu_hadm = dict([(h, cohort[cohort.hadm_id == h].icustay_id.tolist()) for h in lst])
    icu_dct = {}
    for key,val in icu_hadm.items():
        for v in val:
            icu_dct[v] = key
    icustays = sorted(icu_dct.keys())
    
    dfs= [p_bg, p_lab, p_vital, p_uo, p_gcs]
    lsts = [lst, lst, icustays, icustays, icustays]
    cols = [['so2', 'spo2', 'po2', 'pco2', 'fio2_chartevents', 
       'aado2_calc', 'pao2fio2ratio', 'ph', 'baseexcess', 'bicarbonate',
       'totalco2', 'hematocrit', 'hemoglobin', 'methemoglobin',  'calcium', 'temperature', 'potassium',
       'sodium', 'lactate', 'glucose',  'tidalvolume', 'peep'],        
        ['aniongap', 'albumin', 'bilirubin', 'creatinine', 'platelet',
       'ptt', 'inr', 'pt', 'bun', 'wbc'],
         ['heartrate', 'sysbp', 'diasbp', 'meanbp',
       'resprate', 'tempc', 'spo2', 'glucose'],
        ['urineoutput'],
        ['gcs']]
    #variables = (lambda l: [item for sublist in l for item in sublist])(cols)
    #add variables to each timestep
    for k in features.keys():
        tmp = {}
        for t in sorted(features[k])[-48:]:
            tmp[t] = {}
        features[k] = tmp

    for idx in range(len(dfs)):
        for c in cols[idx]:
            print(c)
            #get quintile bins
            bins = pd.qcut(dfs[idx][c], q=5, retbins = True, duplicates = 'drop')[1]
            dct_bins[c] = bins
            #for each admission, for each hourly bin, compile bow vector for features
            for i in progressbar.progressbar(range(len(lsts[idx]))):
                h = lsts[idx][i]
                if len(lst) == len(lsts[idx]):
                    s = dfs[idx][dfs[idx]['hadm_id']==h].set_index('charttime')[c]
                else:
                    s =  dfs[idx][dfs[idx]['icustay_id']==h].set_index('charttime')[c]
                    h = icu_dct[h]
                s = pd.cut(s, bins, labels = False)
                #extubation timeframe
                s = s[(s.index<=dct[hadm_dct[h]]['mv_extub'])]
                #s = s.resample('H').apply(lambda x: np.sum(one_hot(x,6), axis=0) if not pd.isnull(x).all() else np.array([0, 0, 0, 0, 0, 1])).to_dict()
                s = s.resample('H').apply(lambda x: bow_sampler(x, len(bins)-1)).to_dict()
                
                s = dict([(key,val) for key,val in s.items() if key in features[h].keys()])
                if pd.isnull(list(s.values())).all():
                    for t in sorted(features[h].keys()):
                        features[h][t][c] = np.zeros((len(bins)-1), dtype=int)
                else:
                    times = sorted([tt for (tt, val) in s.items() if not np.isnan(val).all()])
                    for t in sorted(features[h].keys()):
                        if t < times[0]:
                            features[h][t][c] = s[times[0]][0]
                        elif t in times:
                            features[h][t][c] = s[t][1]
                        elif t not in s.keys():
                            curr = find_nearest(sorted(s.keys()), t)
                            features[h][t][c] = s[curr][2]
                            s[t] = s[curr]
                        else:
                            prev = find_prev(sorted(s.keys()), t)
                            features[h][t][c] = s[prev][2]
                            s[t] = s[prev]
                    
    return features, dct_bins

def pivot_notes():
    notes = pd.read_csv(path_notes + 'noteevents.csv')
    ### NLP Processing ###
    import nltk
    return 

def pivot_vaso(ref):
    p_vaso = pd.read_csv(path_views + '/drugs/vassopressordurations.csv')
    ## hourly binning ##
    return p_vaso

def build_pivot():
    p_icd, p_drg = pivot_dx()
    #original cohort
    cohort = pd.read_csv(path_views + 'cohorts/icustay_detail.csv')
    cohort = cohort[['subject_id', 'hadm_id', 'icustay_id','intime', 'outtime', 'admittime']]
    #MV patients
    dct = exclude_trach()
    
    ##filter cohort by MV labels##
    cohort = cohort[cohort.subject_id.isin(dct.keys())].sort_values(by=['subject_id', 'admittime', 'intime'])
    labels = pd.DataFrame(dct); labels = labels.T
    #labels.index.name='subject_id'; labels.reset_index(level=0, inplace=True)
    labels = labels[['icustay_id', 'mv_extub']]
    #labels.subject_id = labels.subject_id.astype(int) 
    labels.icustay_id = labels.icustay_id.astype(int)
    #cohort.hadm_id = cohort.hadm_id.astype(int); cohort.icustay_id = cohort.icustay_id.astype(int)
    df = pd.merge(cohort, labels, how = 'outer', on = 'icustay_id')
    df.admittime = pd.to_datetime(df.admittime); df.mv_extub = pd.to_datetime(df.mv_extub)
    df = df[df.intime < df.mv_extub]
    p_bg, p_gcs, p_uo, p_vital, p_lab = pivot_labs()
    p_vaso = pivot_vaso()
    #df = pd.merge(cohort, labels, left_on = ['subject_id', 'hadm_id', 'icustay_id'], right_on = ['subject_id','hadm_id', 'icustay_id'])
    df = df[['subject_id', 'hadm_id', 'icustay_id', 'mv_extub']]
    ##merge on hadm_id
    df = pd.merge(df, p_icd, on = 'hadm_id')
    df = pd.merge(df, p_drg, on = 'hadm_id')
    df = pd.merge(df, p_bg, on= 'hadm_id')
    df.charttime=pd.to_datetime(df.charttime)
    df = df[df.charttime <= df.mv_extub]
    df = pd.merge(df, p_lab, how = 'outer', on='hadm_id')
    df.charttime=pd.to_datetime(df.charttime)
    df = df.drop(columns = 'hadm_id')
    ##merge on icustay
    df = pd.merge(df, p_gcs, on = 'icustay_id')
    df = pd.merge(df, p_uo, on = 'icustay_id')
    df = pd.merge(df, p_vital, on= 'icustay_id')
    df = pd.merge(df, p_vaso, on = 'icustay_id')
    ##merge on charttime
    
    return df


