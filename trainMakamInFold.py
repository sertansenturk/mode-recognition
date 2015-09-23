# coding: utf-8

import sys
sys.path.insert(0, '../ModeTonicEstimation/')

import json
import os
from extras import fileOperations as fo
import numpy as np
from ModeTonicEstimation import Chordia

# I/O
base_dir = '../../experiments/raag-recognition/'
data_dir = os.path.join(base_dir,'data')
experiments_dir = os.path.join(base_dir, 'experiments')
modes = fo.getModeNames(data_dir)
print len(modes)

che = Chordia.Chordia(step_size=10, smooth_factor=15, chunk_size=0, 
					threshold=0.5, overlap=0, frame_rate=128.0/44100)

# indexing
n_exp = 20
n_folds = 12
n_modes = len(modes)  # 10 raagas


for ex_idx in xrange(0,n_exp):
	for fo_idx in xrange(0, n_folds):
		for cur_mode in modes:
			# create load the raag from the experiment & fold corresponding to the input index
			foldFile = os.path.join(experiments_dir, 'exp' + str(ex_idx), 'folds.json')
			with open(foldFile) as f:
			    folds = json.load(f)
			        
			makamTrain_dict = folds['fold'+str(fo_idx)]['train']

			[file_list, tonic_list] = zip(*[(rec['file'], rec['tonic']) for rec in makamTrain_dict
			                                if rec['mode'] == cur_mode])

			train_savefolder = os.path.join(experiments_dir, 'exp' + str(ex_idx), 
			                                'fold' + str(fo_idx), 'train')
			model = che.train(cur_mode, file_list, tonic_list, metric='pcd', 
			                             save_dir = train_savefolder)
			print "Finished training: " + os.path.join(train_savefolder, cur_mode)
