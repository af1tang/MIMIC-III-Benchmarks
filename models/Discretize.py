# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:04:34 2017

@author: af1tang
"""
import sys, pickle
import os.path as path

import csv
import gzip
#import MySQLdb as mysql
import pymysql as mysql
import pandas as pd
from pandas import DataFrame
from pandas.io import sql as transfer


import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import re

from scipy import stats
from collections import Counter
from itertools import combinations
from datetime import date
from datetime import time
from datetime import timedelta
from tempfile import mkdtemp
from sklearn import preprocessing


class Lab:
       
    def __init__(self, df, flags):
        self.df = df
        flags = [str(i) for i in flags]
        if list(set(self.df.FEATURE))[0] in flags:
            self.flag = True
        else: 
            self.flag = False
        
    def is_number(s):
        try:
            float(s)
            return True
        except:
            return False
        
    def is_discrete (self):
        temp = self.df.copy(deep=True)
        temp.VALUE.replace(to_replace='[^0-9]+', value='', inplace = True, regex=True)
        
        unique = list(set(list(temp['VALUE'])))
        digits = [x for x in unique if Lab.is_number(x)]
        if len(digits)/len(unique) > .95: 
            self.discrete= False
        else: 
            self.discrete = True
                                
    
    def clean_list(self):
        if self.flag == True:
            temp = list(set(self.df.FLAG))
            temp = [x for x in temp if x is not None]
            self.df['DISCRETE_VALUE'] = self.df.FLAG.apply(lambda x: 1 if (x in temp) else 0)
      
        else:
            if self.discrete == True:
                #use pandas regex to get rid of ERROR values.
                try:
                    e = ~self.df.VALUE.str.contains(r'(ERROR|FAILURE)', flags = re.IGNORECASE, regex = True)
                    self.df = self.df[e]
                except:
                    pass
                Lab.d_to_d(self)
            
            else:
                Lab.c_to_d(self)
        self.max = self.df.DISCRETE_VALUE.max()
        
    def c_to_d (self):
        self.df['VALUE'] = self.df['VALUE'].convert_objects(convert_numeric=True)
        try:
            try:
                ranked = stats.rankdata(list(self.df['VALUE']))
            except:
                self.df.VALUE = self.df.VALUE.str.split('/').str[0]
                self.df['VALUE'] = self.df['VALUE'].convert_objects(convert_numeric=True)
                ranked = stats.rankdata(list(self.df['VALUE']))
            percentiles = ranked/len(list(self.df['VALUE']))*100
            bins = [0,20,40,60,80,100]
            self.df['DISCRETE_VALUE']= np.digitize(percentiles, bins, right= True)
        
        except:
            self.df['DISCRETE_VALUE'] = np.nan
        
    
    def d_to_d (self): 
        cats = preprocessing.LabelEncoder()
        try:
            self.df['DISCRETE_VALUE'] = cats.fit_transform(self.df['VALUE'])
        except:
            self.df['DISCRETE_VALUE'] = np.nan

    
    
# SCRATCH WORK 
    
    #loop through labs by patients
        #df = pd.read_csv(admissions_doc)
        #subjects = dict(Counter(df["SUBJECT_ID"])) #creates Counter for each unique subject
        #subj = list(subjects.keys())
        #subj = [str(i) for i in subj]
    
        #for s in subj:
        #    sql = "SELECT * from UFM where SUBJECT_ID = '%s' AND TYPE = '%s'" % (s, 'l')
        #    df = pd.read_sql_query(sql=sql, con = conn)

    #regex processing
            #strings = [x for x in self.unique if not is_number(x)]
        #self.values = [float(x) for x in self.values if is_number(x)]
        #if self.discrete == False:
        #    #regex
        #    for s in strings:
        #        r = re.findall('\d+\.\d+', s)
        #    c_to_d
        #else:
        #    d_to_d

        
