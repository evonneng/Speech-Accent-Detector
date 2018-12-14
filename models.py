from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy:
	def __init__(self, model):
		self.model = model
		self.hist = []

	def __call__(self, obs):
		self.hist.append(obs)
		if len(self.hist) > self.model.width:
			self.hist = self.hist[-self.model.width:]
		x = torch.stack(self.hist, dim=0)[None]
		return self.model(x)[0,-1,:]

class Model(nn.Module):
	def __init__(self):
		super().__init__()

		self.fc = nn.Sequential(
				nn.Linear(12, 256),
				nn.ReLU(True),
				nn.Linear(256, 32)
			)

		self.t_cnn = nn.Sequential(
				nn.Conv1d(32, 16, 5),
				nn.ReLU(True),
				nn.Conv1d(16, 16, 5),
				nn.ReLU(True),
				nn.Conv1d(16, 16, 5),
				nn.ReLU(True),
				nn.Conv1d(16, 5, 5)
			)

		self.width = 16

	def forward(self, hist):
		# input (batch_size, sequence_length, features)
		# output (batch_size, sequence_length, 5)
		b, s, f = hist.size()
		hist = hist.view(b * s, f)
		x = self.fc(hist)
		x = x.view(b, s, -1)
		x = x.permute(0, 2, 1)
		x = F.pad(x, (self.width, 0))
		results = self.t_cnn(x)
		results = results.permute(0, 2, 1)
		return results

	def policy(self):
		return Policy(self)