import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from utils import dataset
from utils import rf as setup
from models import *

dirname = os.path.dirname(os.path.abspath(__file__))


def cycle(seq):
    while True:
        for elem in seq:
            yield elem


def train(max_iter, numlang, languages, batch_size=10):
    '''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

    """
	Load the training data
	"""

    data = {}
    iterator = {}
    for i in range(numlang):
        data[languages[i]] = dataset.load(os.path.join(dirname, 'recordings_wav', languages[i]), languages[i], num_workers=0, crop=100, batch_size=batch_size)
        iterator[languages[i]] = cycle(data[languages[i]])

    try:
        model = Model()
        model.load_state_dict(torch.load(os.path.join(dirname, 'model.th'), map_location=lambda storage, loc: storage))
    except:
        model = Model()
    print (model)

    print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    # If your model does not train well, you may swap out the optimizer or change the lr

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-4)

    loss = nn.BCEWithLogitsLoss()
    for t in range(max_iter):
        for idx in range(numlang):
            batch_mfcc = next(iterator[languages[i]]).float()
            batch_labels = np.tile(np.array([1.0 if i == idx else 0.0 for i in range(len(languages))]), (batch_mfcc.shape[0], batch_mfcc.shape[1], 1))
            batch_labels = torch.from_numpy(batch_labels).float()
            model.train()
            optimizer.zero_grad()
            model_outputs = model(batch_mfcc)
            t_loss_val = loss(model_outputs, batch_labels)
            t_loss_val.backward()
            optimizer.step()

            if t % 10 == 0:
                print('[%5d - %s] t_loss = %f' % (t, languages[i], t_loss_val))

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'model.th'))  # Do NOT modify this line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--max_iter', type=int, default=5000)
    parser.add_argument('-l', '--numlang', type=int, default=5)
    args = parser.parse_args()

    print("num langs: {}".format(args.numlang))

    print ('[I] Start training')
    train(args.max_iter, args.numlang, ['english', 'spanish', 'french', 'arabic', 'russian'])
    print ('[I] Training finished')