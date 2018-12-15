import librosa
import sys
import os
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from prosody import get_dynamic_features, get_static_features

RATE = 8000
N_MFCC = 12

NUM_FEATURES_MFCC = 3000
NUM_FEATURES_PROSODIC_S = 38

def get_wav(filename):
	'''
	filename: file name to load
	return: numpy array down-sampled wav file
	'''
	y, sr = librosa.load('{}'.format(filename))
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
	print("currently processing audio file: {}".format(filename))

	wav = get_wav(filename)
	mfcc = to_mfcc(wav).flatten()

	return mfcc[:NUM_FEATURES_MFCC]

def add_dir_to_df(dir_name, X, mfcc=True):
	'''
	dir_name: the directory to add files to dataframe
	X: features dataframe to add to
	'''
	filenames = map(lambda rel_path: dir_name + "/" + rel_path, os.listdir(dir_name))

	if mfcc:
		features = list(map(get_mfcc_from_filename, filenames))
	else:
		features = list(map(get_static_features, filenames))

	new_rows = pd.DataFrame(features, columns=X.columns)
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


def load_dataframe(mfcc=True):
	if mfcc:
		X = pd.DataFrame(columns=list(range(NUM_FEATURES_MFCC)))
	else:
		X = pd.DataFrame(columns=list(range(NUM_FEATURES_PROSODIC_S)))

	y = pd.DataFrame(columns=["y"])

	dirs = ["../recordings_wav/english", "../recordings_wav/spanish", "../recordings_wav/arabic"]#, "../recordings_wav/french"]
	# dirs = ["../recordings_wav/english", "../recordings_wav/french"]

	for label, d in enumerate(dirs):
		if mfcc:
		    X = add_dir_to_df(d, X, mfcc=True)
		else:
			X = add_dir_to_df(d, X, mfcc=False)
		y = add_category_to_labels(d, label, y)

	y = y.values.ravel()

	print("done making dataframe, now making model...")
	return X, y

def check_cross_val_score(X, y):
	clf = RandomForestClassifier(n_estimators=100)
	return cross_val_score(clf, X, y, cv=3).mean()

def main():
	mfcc = sys.argv[1]

	if mfcc == "true":
		X, y = load_dataframe(mfcc=True)
	else:
		X, y = load_dataframe(mfcc=False)

	print(check_cross_val_score(X, y))

if __name__ == '__main__':
	main()



