import librosa
import sys
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

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

def add_dir_to_df(dir_name, X):
	'''
	dir_name: the directory to add files to dataframe
	X: features dataframe to add to
	'''
	filenames = os.listdir(dir_name)
	print(len(filenames))

	for i in range(X.shape[0], X.shape[0] + len(filenames)):
		filename = filenames[i - X.shape[0]]

		wav = get_wav(dir_name + "/" + filename)
		mfcc = to_mfcc(wav).flatten()

		X.loc[i] = mfcc[:NUM_FEATURES]


def add_category_to_labels(dir_name, label, y):
	'''
	dir_name: the directory to add files to dataframe
	label: label of new category
	y: predictions dataframe to add to
	'''
	files = os.listdir(dir_name)
	print(len(files))

	num_rows_to_add = len(files)
	for i in range(y.shape[0], num_rows_to_add + y.shape[0]):
		y.loc[i] = label


if __name__ == '__main__':
	X = pd.DataFrame(columns=list(range(NUM_FEATURES)))
	y = pd.DataFrame(columns=["y"])

	add_dir_to_df("../recordings_wav/c", X)
	add_category_to_labels("../recordings_wav/c", '0', y)

	print(X.shape)
	print(y.shape)

	add_dir_to_df("../recordings_wav/d", X)
	add_category_to_labels("../recordings_wav/d", '1', y)

	print(X.shape)
	print(y.shape)

	clf = LogisticRegression(random_state=0, solver='lbfgs',
                          multi_class='multinomial').fit(X, y)

	test_val = to_mfcc(get_wav("../recordings_wav/gujarati1.wav")).flatten()
	prediction = clf.predict(test_val)

	print(prediction)








