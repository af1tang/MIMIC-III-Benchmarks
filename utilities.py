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

def bow_to_ohv(dct):
    '''converts bag of words to one-hot format.
    dct: {hadm_id: (X, y)}
    '''
    xy = {}
    for h in dct.keys():
        X, y = [], dct[h][1]
        for t in dct[h][0]:
            X.append(np.array( [(lambda x: 1 if xx > 0 else 0)(xx) for xx in t] ) )
        X = np.array(X)
        xy[h] = (X,y)
    return xy

def bow_sampler(x, size):
    if not pd.isnull(x).all():
        bow = np.sum(one_hot(x, size), axis=0) 
        bow = np.array([(lambda x: 1 if x >0 else 0)(xx) for xx in bow])
        first = one_hot(x,size)[0]
        last = one_hot(x,size)[-1]
        return [first, bow, last]
    else:
        return np.nan

def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

### word2vec Dictionary processing ### 
def normalize_dct(d):
    normalized_d = {}
    for k,v in d.items():
        norm_v = v / np.sqrt((v ** 2).sum(-1))
        normalized_d[k] = norm_v
    return normalized_d

def negative_sampling(codes, dct, training_mode = False):
    if training_mode:
        pos = [(item, 1) for item in dct.keys() if item in codes]                
        topn = max(min(5, len(pos)), 10)
        ez = [(key,0) for key,item in most_similar(dct, lst = pos, topn= topn, reverse = True)]
        hard = [(key,0) for key,item in most_similar(dct, lst = pos, topn= topn)]
        items = shuffle(pos+ ez + hard)
    else:
        pos = [(item, 1) for item in dct.keys() if item in codes]                
        neg = [(item, 0) for item in dct.keys() if item not in codes]
        items = shuffle(pos + neg)
    return items

def std_split(dct, dct_notes, dct_icd):
    '''dct: {h: (x_t, notes, codes)}
    '''
    def RepresentsInt(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
    random.seed(119304913)
    random_state = 99991
    ss = StratifiedShuffleSplit(n_splits = 1, test_size = .25, random_state = random_state)
    
    _, _, y = zip(*dct.values())
    y = np.array(y)
    
    #get unique icd counts, set y to get stratified splits
    icd_keys, icd_counts = np.unique(flatten(y), return_counts = True)
    icd_dct = [(k,v) for (k,v) in list(zip(icd_keys, icd_counts)) if ((v >= 250) & RepresentsInt(k))]
    icd_keys, icd_counts = zip(*icd_dct)
    icd_dct = sorted(icd_dct, key = lambda x: x[1], reverse = True)
    icd_dct = dict([(v,k) for k,v in enumerate([i[0] for i in icd_dct])])
    #remake features
    features = {}
    for h in dct.keys():
        x, z, yy = dct[h][0], dct[h][1], dct[h][2]
        yy = np.array([item for item in yy if item in icd_keys])
        if len(yy)>1:
            features[h] = (x, z, yy)
    X, Z, y = zip(*features.values())
    X, Z, y = np.array(X), np.array(Z), np.array(y)       
    #y = np.array([np.array([yy for yy in y[i] if yy in icd_keys]) for i in range(len(y))])

    #y_ohv for splitting
    y_ohv = np.array([np.sum(one_hot([icd_dct[item] for item in list(set(yy))], len(icd_keys)), axis=0) for yy in y])
    
    #get unique counts of note features 
    unique, count = np.unique(flatten(Z), return_counts = True)
    items = [(k,v) for (k,v) in list(zip(unique, count)) if v >= 100]
    items = sorted(items, key = lambda x: x[1], reverse = True)
    items = items[0:500]
    unique, count = zip(*items)
    
    #remake dictionaries
    note_dct, icd_dct = dict([(k,v) for k,v in dct_notes.items() if k in unique]), dict([ (k,v) for k,v in dct_icd.items() if k in icd_keys])
    #normalize dictionaries
    #note_dct, icd_dct = normalize_dct(note_dct), normalize_dct(icd_dct)

    train_index, test_index = list(ss.split(X, y_ohv[:, -1]))[0]
    X_tr, X_te = X[train_index], X[test_index]
    Z_tr, Z_te = Z[train_index], Z[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    #normalize
    #X_tr = np.array([np.array([(xxx+2.)/4. for xxx in xx]) for xx in  X_tr])
    #X_te = np.array([np.array([(xxx+2.)/4. for xxx in xx]) for xx in  X_te])
    return [(X_tr, X_te), 
            (Z_tr, Z_te), (y_tr, y_te),
            note_dct, icd_dct]
    
def most_similar(word_dct, lst =[], topn=10, reverse=False):
    """word_dct: {w: vec}, normalized
    """
    
    if isinstance(lst, string_types):
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        lst = [lst]
    lst = [
        (word, 1.0) if isinstance(word, string_types + (np.ndarray,)) else word
        for word in lst
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in lst:
        mean.append(weight * word_dct[word])
        if word in word_dct.keys():
            all_words.add(word)
            
    mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

    results = [(k, np.dot(v, mean)) for k,v in word_dct.items() if k not in all_words]
    if reverse:
        results = np.array(sorted(results, key=lambda x: x[1], reverse=False))      
    else:
        results = np.array(sorted(results, key=lambda x: x[1], reverse=True))
 
    return results[:topn]

### Progressbar tools ### 
def make_widget():
    widgets = [progressbar.Percentage(), ' ', progressbar.SimpleProgress(), ' ', 
                                 progressbar.Bar(left = '[', right = ']'), ' ', progressbar.ETA(), ' ', 
                                 progressbar.DynamicMessage('LOSS'), ' ',  progressbar.DynamicMessage('PREC'), ' ',
                                 progressbar.DynamicMessage('REC')]
    bar = progressbar.ProgressBar(widgets = widgets)
    return bar


### Find nearest timestamps ###
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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
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

#### Pickling Tools ####

def large_save(dct, file_path):
    '''dct: {k: v}
    '''
    lst = sorted(dct.keys())
    chunksize =10000
    #chunk bytes
    bytes_out= bytearray(0)
    for idx in range(0, len(lst), chunksize):
        bytes_out += pickle.dumps(dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]]))
    with open(file_path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), chunksize):
                f_out.write(bytes_out[idx:idx+chunksize])
    #split files
    for idx in range(0, len(lst), chunksize):
        chunk = dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]])
        with open(file_path+'features_'+str(idx+chunksize), 'wb') as f_out:
            pickle.dump(chunk, f_out, protocol=2)

def large_read(file_path):
    import os.path
    bytes_in = bytearray(0)
    max_bytes = int(1e5)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data

#### Score Reporting ####
def test_uni(y_te, yhat):
    yhat, y_te = np.array(yhat).ravel(), np.array(y_te).ravel()
    fpr, tpr, thresholds = roc_curve(y_te, yhat)
    roc_auc = auc(fpr, tpr)
    prec, rec, thresholds = precision_recall_curve(y_te, yhat)
    pr_auc = average_precision_score(y_te , yhat)
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.abs(prec- rec))
    optimal_threshold = thresholds[optimal_idx]
    yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
    yhat=[int(i) for i in yhat]
    f1=f1_score(y_te,yhat)
    #prec= precision_score(y_te, yhat)
    #rec = recall_score(y_te, yhat)
    return roc_auc,f1, pr_auc

def test_multi(y_te, ypred):
    ypred, y_te = np.array(ypred), np.array(y_te)
    precs, recs, aucs = [], [], []
    for i in range(ypred.shape[1]):
        y_true, yhat = y_te[:, i], ypred[:, i]
        fpr, tpr, thresholds = roc_curve(y_true, yhat)
        aucs.append(auc(fpr, tpr))
        prec, rec, thresholds = precision_recall_curve(y_true, yhat)
        optimal_idx = np.argmin(np.abs(prec-rec))
        #optimal_idx = np.argmin(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
        yhat=[int(i) for i in yhat]
        precs.append(precision_score(y_true, yhat))
        recs.append(recall_score(y_true, yhat))
    return precs, recs, aucs

def score_uni(y_te, yhat):
    yhat, y_te = np.array(yhat), np.array(y_te)
    yhat, y_te = yhat.ravel(), y_te.ravel()
    fpr, tpr, thresholds = roc_curve(y_te, yhat)
    roc_auc = auc(fpr, tpr)
    prec, rec, thresholds = precision_recall_curve(y_te, yhat)
    #optimal_idx = np.argmax(tpr - fpr)
    optimal_idx = np.argmin(np.abs(prec- rec))
    optimal_threshold = thresholds[optimal_idx]
    yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
    yhat=[int(i) for i in yhat]
    f1=f1_score(y_te,yhat)
    return roc_auc, f1

def score_multi(y_te, ypred):
    ypred, y_te = np.array(ypred), np.array(y_te)
    precs, recs, aucs, f1s, optimals= [], [], [],[], []
    for idx in range(ypred.shape[1]):
        y_true, yhat = y_te[:, idx], ypred[:, idx]
        #thresholds = np.histogram(yhat, bins=20)[1]
        fpr, tpr, thresholds = roc_curve(y_true, yhat)
        aucs.append(auc(fpr, tpr))
        f1 =[]
        for thresh in thresholds:
            yhat[yhat>=thresh] = 1
            yhat[yhat<thresh] = 0
            f1.append(f1_score(y_true, yhat))
        optimal_idx = np.argmax(f1)
        thresh = thresholds[optimal_idx]
        yhat[yhat>=thresh] = 1;
        yhat[yhat<thresh] = 0
        precs.append(precision_score(y_true, yhat))
        f1s.append(f1[optimal_idx])
        recs.append(recall_score(y_true, yhat))
        optimals.append(thresh)
    return aucs, precs, recs, f1s, optimals

### Model Interpretability ###
def get_activations(model, inputs, name ='memory_output'):
    '''inputs: [x_te, icd_9, memory]'''
    layer = Model(inputs=model.input, outputs=model.get_layer(name).get_output_at(1)[1])
    attention = layer.predict(inputs)
    return attention

def plot_timeseries(feature_dct, x,
                    lst = [6,9, 34, 38, 27, 12, 15, 23, 31, 35, 16, 33]):
    '''lst : [bun, diasbp, gcs, glucose, heartrate, pao2fio2ratio, 
    platelet, resprate, spo2, sysbp, tempc, urineoutput]'''
    x = x.reshape(24,40)
    x = x.T
    x = dict([(feature_dct[k], v) for k,v in enumerate(x) if k in lst])
    df = pd.DataFrame(x)
    #df.columns = ['Standard Deviation']
    ax = df.plot(title = 'Subset of Timeseries Features for Test Patient')
    ax.set_xlabel("Hours")
    ax.set_ylabel("Standard Deviation")
    plt.show()

def plot_attention(umls_dct, attention):
    attention = attention.reshape(500,1)
    attention = attention[0:20]
    attention_vec = dict([ (umls_dct[k], v) for k,v in enumerate(attention)])
    df = pd.DataFrame(attention_vec).T
    df.columns = ['attention weights']
    df.plot(kind = 'bar', title = 'Attention over top 20 common UMLS keywords')
    plt.show()

def plot_dx(dx_dct, y, name = 'Predicted'):
    y = y.reshape(148, 1)
    #y = dict([ (dx_dct[k], v) for k,v in enumerate(y)])
    df = pd.DataFrame(y)
    df.columns = [name]
    ax = df.plot(kind = 'bar', title = 'Predicted ICD-9 Codes')
    ax.set_xlabel("ICD-9 Codes")
    ax.set_ylabel("Activations")
    x_axis = ax.axes.get_xaxis()
    x_axis.set_ticks([])
    plt.show()
