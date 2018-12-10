import librosa

def get_wav(filenames):
	'''
	filenames: list of file names to load
	return: numpy array down-sampled wav file
	'''
	y, sr = librosa.load('../audio/{}.wav'.format(filenames))
	return librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True)

def to_mfcc(wav):
	'''
	wav: numpy array down-sampled wav file
	return: 2d numpy array of mfcc
	'''
	return librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)


if __name__ == '__main__':
	filename = sys.argv[1]
	wav = get_wav(filename)
	mfcc = to_mfcc(wav)
