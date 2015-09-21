
# coding: utf-8

# In[1]:

import sys
sys.path.insert(0, '../ModeTonicEstimation/')

import json
import os
from extras import foldGeneration
from extras import fileOperations as fo
import numpy as np
from sklearn import cross_validation


# In[2]:

# I/O
base_dir = '../../experiments/raag-recognition/'
data_dir = os.path.join(base_dir,'data')
experiments_dir = os.path.join(base_dir, 'experiments')
modes = fo.getModeNames(data_dir)

n_exp = 20
n_folds = 14


# In[3]:

# get the data into appropriate format
[pitch_paths, pitch_base, pitch_fname] = fo.getFileNamesInDir(data_dir, '.pitch')
tonic_paths = [os.path.splitext(p)[0] + '.tonic' for p in pitch_paths]
mode_labels = []
for p in pitch_base:
    for r in modes:
        if r in p:
            mode_labels.append(r)
            


# In[4]:

# make the data a single dictionary for housekeeping
data = []
for p, f, t, r in zip(pitch_paths, pitch_fname, tonic_paths, mode_labels):
    data.append({'file':p, 'name':os.path.splitext(f)[0],
               'tonic':float(np.loadtxt(t)), 'mode':r})


# In[5]:

# create 20 stratified 14 fold 
mode_idx = [modes.index(m) for m in [d['mode'] for d in data]]

for nn in xrange(0,n_exp):
    skf = cross_validation.StratifiedKFold(mode_idx, n_folds=n_folds)

    folds = dict()
    for ff, fold in enumerate(skf):
        folds['fold' + str(ff)] = {'train': [], 'test': []}
        for tr_idx in fold[0]:
            folds['fold' + str(ff)]['train'].append(data[tr_idx])
        for te_idx in fold[1]:
            folds['fold' + str(ff)]['test'].append(data[te_idx])

    exp_dir = os.path.join(experiments_dir, 'exp' + str(nn))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    fold_savefile = os.path.join(exp_dir, 'folds.json')
    with open(fold_savefile, 'w') as f:
        json.dump(folds, f, indent=2)
    
    print "Created the folds for Experiment " + str(nn)


# In[ ]:



