# coding: utf-8


import sys
sys.path.insert(0, '../ModeTonicEstimation/')

import json
import os
from extras import fileOperations as fo
import numpy as np
from ModeTonicEstimation import Chordia
from ModeTonicEstimation import Evaluator as ev


# I/O
input_num = int(sys.argv[1])-1

base_dir = '../../experiments/raag-recognition/'
data_dir = os.path.join(base_dir,'data')
experiments_dir = os.path.join(base_dir, 'experiments')
modes = fo.getModeNames(data_dir)

evaluator = ev.Evaluator()
che = Chordia.Chordia(step_size=10, smooth_factor=15, chunk_size=120, 
                      threshold=0.5, overlap=0, frame_rate=128.0/44100)


# indexing
n_exp = 20
n_folds = 12

mode_idx = np.unravel_index(input_num, [n_exp, n_folds])

trainFolder = os.path.join(experiments_dir, 'exp' + str(mode_idx[0]), 
                           'fold' + str(mode_idx[1]), 'train')
resultFile = os.path.join(experiments_dir, 'exp' + str(mode_idx[0]), 
                          'fold' + str(mode_idx[1]), 'results.json')


# create load the raag from the experiment & fold corresponding to the input index
foldFile = os.path.join(experiments_dir, 'exp' + str(mode_idx[0]), 'folds.json')


with open(foldFile) as f:
    folds = json.load(f)
        
makamTestFiles = folds['fold'+str(mode_idx[1])]['test']


# Raag Recognition
results = []
for rec in makamTestFiles:
    rec['pitch'] = np.loadtxt(rec['file'])[:,1]
    res = che.estimate(rec['pitch'], mode_names=modes, mode_dir=trainFolder, 
                       est_tonic=False, est_mode=True, distance_method='bhat',
                       metric='pcd', ref_freq=rec['tonic'], 
                       equalSamplePerMode=True)[0]

    # evaluate
    results.append(evaluator.mode_evaluate(rec['file'], res, rec['mode']))

print('exp' + str(mode_idx[0]) + '_fold' + str(mode_idx[1]) + ":" 
      " " + str(100*np.mean([r['mode_eval'] for r in results])) + '%')

with open(resultFile, 'w') as r:
    json.dump(results, r, indent=4)

