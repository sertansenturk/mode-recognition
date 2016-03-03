# -*- coding: utf-8 -*-
import numpy as np
import sys
import json
import os

from os import path
sys.path.insert(0, './../')
import BozkurtEstimation as be
import ModeFunctions as mf

def run(distance_inp, training_i):
	###Experiment Parameters-------------------------------------------------------------------------
	distance = distance_inp
	training_idx = training_i
	rank = 10
	fold_list = np.arange(1,11)
	distance_list = ['intersection', 'corr', 'manhattan', 'bhat', 'euclidean', 'l3']
	makam_list = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 
				  'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 
				  'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 
				  'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']

	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DATA FOLDER INIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	data_folder = '../../../Makam_Dataset/Pitch_Tracks/'
	#data_folder = '../../../test_datasets/turkish_makam_recognition_dataset/data/' #sertan desktop local
	#data_folder = '../../../experiments/turkish_makam_recognition_dataset/data/' # hpc cluster


	# folder structure
	experiment_dir = './BozkurtExperiments' # assumes it is already created

	#chooses which training to use 
	training_dir = os.path.join(experiment_dir, 'Training' + str(training_idx))
	modePath = os.path.join(training_dir, 'Mode')
	if not os.path.exists(modePath):
		os.makedirs(modePath)

	# get the training experient/fold parameters 
	with open(os.path.join(training_dir, 'parameters.json'), 'r') as f:
		cur_params = json.load(f)
		f.close()

	done_dists = next(os.walk(modePath))[2]
	done_dists = [d[:-5] for d in done_dists]
	if (distance in done_dists):
		print 'Already done ' + distance
		return

	print 'Computing ' + distance
	cent_ss = cur_params['cent_ss']
	smooth_factor = cur_params['smooth_factor']
	distribution_type = cur_params['distribution_type']
	chunk_size = cur_params['chunk_size']

	# instantiate makam estimator for training
	estimator = be.BozkurtEstimation(cent_ss=cent_ss, smooth_factor=smooth_factor, 
		                             chunk_size=chunk_size)

	# load annotations; the tonic values will be read from here
	with open('annotations.json', 'r') as f:
		annot = json.load(f)
		f.close()

	output = dict()
	for fold in fold_list:
		output['Fold' + str(fold)] = []
		fold_dir = os.path.join(training_dir, 'Fold' + str(fold))
		
		# load the current fold to get the test recordings
		with open((os.path.join('./Folds', 'fold_' + str(fold) + '.json')), 'r') as f:
			cur_fold = json.load(f)['test']
			f.close()

		# retrieve annotations of the training recordings
		for makam_name in makam_list:

			# just for checking the uniqueness of test recordings
			with open(os.path.join(fold_dir, makam_name + '.json')) as f:
				makam_recordings = json.load(f)[0]['source']
				f.close()

			# divide the training data into makams
			makam_annot = [k for k in cur_fold if k['makam']==makam_name]
			pitch_track_dir = os.path.join(data_folder, makam_name)

			# load the annotations for testing data; it will be only used for 
			# makam recognition (with annotated tonic)
			for i in makam_annot:
				for j in annot:
					# append the tonic of the recordıng from the relevant annotation
					if(i['mbid'] == j['mbid']):
						i['tonic'] = j['tonic'] 
						break

			#actual estimation
			for recording in makam_annot:
				
				#check if test recording was use in training
				if (recording['mbid'] + '.pitch' in makam_recordings):
					raise ValueError(('Unique-check Failure. ' + recording['mbid']))

				pitch_track = mf.load_track(txt_name=(recording['mbid'] + '.pitch'), 
					                        txt_dir=pitch_track_dir)

				cur_out = estimator.estimate(pitch_track, mode_names=makam_list,
					        est_tonic=False, est_mode=True, rank=rank,
					        distance_method=distance, tonic_freq=recording['tonic'],
					        metric=distribution_type, mode_in=fold_dir)

				output[('Fold' + str(fold))].append({'mbid':recording['mbid'], 'tonic_estimation':cur_out})
	with open(os.path.join(modePath, distance + '.json'), 'w') as f:
		json.dump(output, f, indent=2)
		f.close()
