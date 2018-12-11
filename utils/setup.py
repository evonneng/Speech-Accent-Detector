import librosa
import sys
import os
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

RATE = 8000
N_MFCC = 12

NUM_FEATURES = 3000

def get_wav(filename):
	'''
	filename: file name to load
	return: numpy array down-sampled wav file
	'''
	y, sr = librosa.load('../audio/{}'.format(filename))
	return librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True)

def to_mfcc(wav):
	'''
	wav: numpy array down-sampled wav file
	return: 2d numpy array of mfcc
	'''
	return librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)

def get_mfcc_from_filename(filename):
	'''

	'''
	wav = get_wav(filename)
	mfcc = to_mfcc(wav).flatten()

	return mfcc[:NUM_FEATURES]

def add_dir_to_df(dir_name, X):
	'''
	dir_name: the directory to add files to dataframe
	X: features dataframe to add to
	'''

	filenames = map(lambda rel_path: dir_name + "/" + rel_path, os.listdir(dir_name))
	mfccs = list(map(get_mfcc_from_filename, filenames))

	new_rows = pd.DataFrame(mfccs, columns=X.columns)
	return pd.concat([X, new_rows])


def add_category_to_labels(dir_name, label, y):
	'''
	dir_name: the directory to add files to dataframe
	label: label of new category
	y: predictions dataframe to add to
	'''
	files = os.listdir(dir_name)
	num_rows_to_add = len(files)

	row_labels = pd.DataFrame(np.ones(num_rows_to_add) * label, columns=y.columns)
	return pd.concat([y, row_labels])

if __name__ == '__main__':
	X = pd.DataFrame(columns=list(range(NUM_FEATURES)))
	y = pd.DataFrame(columns=["y"])

	dirs = ["../recordings_wav/hindi", "../recordings_wav/cantonese"]

	for label, d in enumerate(dirs):
		X = add_dir_to_df(d, X)
		y = add_category_to_labels(d, label, y)
	y = y.values.ravel()


	print("done making dataframe, now making model...")
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

	clf.fit(X, y)
	test_val = get_mfcc_from_filename("../recordings_wav/mandarin1.wav").reshape(1, -1)
	prediction = clf.predict_proba(test_val)

	print(prediction)



