from models import *
import os, torch
import argparse, pickle
from utils import setup
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
model = Model()
model.load_state_dict(torch.load(os.path.join(dirname, 'model.th'), map_location=lambda storage, loc: storage))
languages = ['arabic', 'english', 'french', 'spanish', 'russian']

def get_probs(mfcc):
	logits = model(torch.from_numpy(mfcc).float().unsqueeze(0))
	logits = logits.detach().numpy()
	logits = np.mean(np.squeeze(logits), axis=0)
	return 1. / (1. + np.exp(-logits))

def evaluate(filename):
	if filename is not None:
		print('Loading in WAV file')
		wav = setup.get_wav(filename)
		print('Running MFCC')
		mfcc = np.transpose(setup.to_mfcc(wav))
		print('Running Model')
		probs = get_probs(mfcc)
		print('Probabilities:')
		for i, language in enumerate(languages):
			print(language, probs[i])
		print('Highest Probability:', languages[np.argmax(probs)])
	else:
		print('Starting')
		total_correct = 0
		total = 0
		for idx, language in enumerate(languages):
			language_dir = os.path.join(dirname, 'recordings_wav', language)
			files = os.listdir(language_dir)
			language_correct = 0
			language_total = 0
			for f in files:
				wav = setup.get_wav(os.path.join(language_dir, f))
				mfcc = np.transpose(setup.to_mfcc(wav))
				probs = get_probs(mfcc)
				prediction = np.argmax(probs)
				language_total += 1
				total += 1
				if prediction == idx:
					language_correct += 1
					total_correct += 1
			print(language + ': %d/%d (%4f)' % (language_correct, language_total, language_correct / language_total))
		print('Overall: %d/%d (%4f)' % (total_correct, total, total_correct / total))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename')
	args = parser.parse_args()
	evaluate(args.filename)