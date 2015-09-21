# coding: utf-8

import sys
sys.path.insert(0, '../ModeTonicEstimation/')

import json
import os
from extras import foldGeneration
from extras import fileOperations as fo
import numpy as np
from ModeTonicEstimation import Chordia

# I/O
base_dir = '../../experiments/raag-recognition/'
data_dir = os.path.join(base_dir,'data')
experiments_dir = os.path.join(base_dir, 'experiments')
modes = fo.getModeNames(data_dir)

input_num = int(sys.argv[1])-1

che = Chordia.Chordia(step_size=10, smooth_factor=15, chunk_size=120, 
					threshold=0.5, overlap=0, frame_rate=128.0/44100)

# indexing
n_exp = 20
n_folds = 12
n_modes = len(modes)  # 10 raagas

mode_idx = np.unravel_index(input_num, [n_exp, n_folds, n_modes])

# create load the raag from the experiment & fold corresponding to the input index
foldFile = os.path.join(experiments_dir, 'exp' + str(mode_idx[0]), 'folds.json')
with open(foldFile) as f:
    folds = json.load(f)
        
makamTrain_dict = folds['fold'+str(mode_idx[1])]['train']

cur_mode = modes[mode_idx[2]]        
[file_list, tonic_list] = zip(*[(rec['file'], rec['tonic']) for rec in makamTrain_dict
                                if rec['mode'] == cur_mode])

train_savefolder = os.path.join(experiments_dir, 'exp' + str(mode_idx[0]), 
                                'fold' + str(mode_idx[1]), 'train')
model = che.train(cur_mode, file_list, tonic_list, metric='pcd', 
                             save_dir = train_savefolder)


