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

# from prosody import get_dynamic_features, get_static_features

RATE = 8000
N_MFCC = 12

NUM_FEATURES_MFCC = 3000
NUM_FEATURES_PROSODIC_S = 38
NUM_FEATURES_PROSODIC_D = 611

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
	print("currently processing mfcc from audio file: {}".format(filename))

	wav = get_wav(filename)
	mfcc = to_mfcc(wav).flatten()

	return mfcc[:NUM_FEATURES_MFCC]

def get_combined_features(filename):
	print("getting combined prosodic features")
	return np.concatenate(get_static_features(filename), get_dynamic_features(filename))

def add_dir_to_df(dir_name, X, mfcc=True, dynamic=False):
	'''
	dir_name: the directory to add files to dataframe
	X: features dataframe to add to
	'''
	filenames = map(lambda rel_path: dir_name + "/" + rel_path, os.listdir(dir_name))

	if mfcc:
		features = list(map(get_mfcc_from_filename, filenames))
	else:
		if not dynamic:
			features = list(map(get_static_features, filenames))
		else:
			features = list(map(get_dynamic_features, filenames))

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


# def load_dataframe(mfcc=True, dynamic=False):
# 	if mfcc:
# 		X = pd.DataFrame(columns=list(range(NUM_FEATURES_MFCC)))
# 	else:
# 		if not dynamic:
# 			X = pd.DataFrame(columns=list(range(NUM_FEATURES_PROSODIC_S)))
# 		else:
# 			X = pd.DataFrame(columns=list(range(NUM_FEATURES_PROSODIC_D)))

# 	y = pd.DataFrame(columns=["y"])

# 	dirs = ["../recordings_wav/english", "../recordings_wav/spanish"]# "../recordings_wav/arabic", "../recordings_wav/french", "../recordings_wav/russian"]

# 	for label, d in enumerate(dirs):
# 		X = add_dir_to_df(d, X, mfcc=mfcc, dynamic=dynamic)
# 		y = add_category_to_labels(d, label, y)

# 	y = y.values.ravel()

# 	X.replace(np.inf, np.nan)
# 	X = X.fillna(X.mean())

# 	print("done making dataframe, now making model...")
# 	return X, y

def check_cross_val_score(X, y):
	clf = RandomForestClassifier(n_estimators=100)
	cvs = cross_val_score(clf, X, y, cv=3)
	return cvs.mean()

def main():
	mfcc = sys.argv[1]

	try:
		dynamic = sys.argv[2]
	except:
		dynamic = "false"

	if mfcc == "true":
		test_increasing_languages(True, False)
	else:
		if dynamic == "false":
			test_increasing_languages(False, False)
		else:
			test_increasing_languages(False, True)

def add_lang(d, mfcc, dynamic, X, y, label):
	X = add_dir_to_df(d, X, mfcc=mfcc, dynamic=dynamic)
	y = add_category_to_labels(d, label, y)

	X.replace(np.inf, np.nan)
	X = X.fillna(X.mean())

	return X, y

def write_score(mfcc, dynamic, X, y, num_langs):
	print(X)
	cv_score = check_cross_val_score(X, y.values.ravel())

	print("writing score {} to file".format(str(cv_score)))

	file = open("{}{}output.txt".format(mfcc, dynamic), "a+")
	file.write("mfcc={}, dynamic_features={}, num_langs={}, cv_score={}".format(mfcc, dynamic, num_langs, cv_score))
	file.close()

def test_increasing_languages(mfcc, dynamic):
	dirs = ["../recordings_wav/english", "../recordings_wav/spanish"]
	# dirs = ["../recordings_wav/greek", "../recordings_wav/hindi"]
	num_langs = len(dirs)

	if mfcc:
		X = pd.DataFrame(columns=list(range(NUM_FEATURES_MFCC)))
	else:
		if not dynamic:
			X = pd.DataFrame(columns=list(range(NUM_FEATURES_PROSODIC_S)))
		else:
			X = pd.DataFrame(columns=list(range(NUM_FEATURES_PROSODIC_D)))

	y = pd.DataFrame(columns=["y"])

	for label, d, in enumerate(dirs):
		X = add_dir_to_df(d, X, mfcc=mfcc, dynamic=dynamic)
		y = add_category_to_labels(d, label, y)

	X.replace(np.inf, np.nan)
	X = X.fillna(X.mean())
	# y = y.values.ravel()
	write_score(mfcc, dynamic, X, y, num_langs)
	print("X.shape after {} langs: {}".format(str(num_langs), str(X.shape)))

	# keep adding languages
	d = "../recordings_wav/french"
	# d = "../recordings_wav/tagalog"
	label = 2
	num_langs += 1

	X, y = add_lang(d, mfcc, dynamic, X, y, label)
	write_score(mfcc, dynamic, X, y, num_langs)
	print("X.shape after {} langs: {}".format(str(num_langs), str(X.shape)))

	d = "../recordings_wav/arabic"
	label = 3
	num_langs += 1

	X, y = add_lang(d, mfcc, dynamic, X, y, label)
	write_score(mfcc, dynamic, X, y, num_langs)

	d = "../recordings_wav/russian"
	label = 4
	num_langs += 1

	X, y = add_lang(d, mfcc, dynamic, X, y, label)
	write_score(mfcc, dynamic, X, y, num_langs)


if __name__ == '__main__':
	main()



