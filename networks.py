import random

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 


class Actor(nn.Module):

	""" Actor Model """
	def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
		super(Actor, self).__init__()
	"""Initialize parameters and build model.
	Params
	======
	    state_size (int)  : Dimension of each state
	    action_size (int) : Dimension of each action
	    seed (int): Random seed
	"""
	self.seed = torch.manual_seed(seed)
	self.fc1 = nn.Linear(state_size, action_size)
	self.bn1 = nn.BatchNorm1d(fc1_units)
	self.fc2 = nn.Linear(fc1_units, fc2_units)
	self.bn2 = nn.Linear(fc2_units)
	self.fc3 = nn.Linear(fc2_units, action_size)
	self.reset_parameters()

	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state)
		x = F.relu(self.fc1(state))
		x_norm = self.bn1(x)
		x = F.relu(self.fc2(x_norm))
		x_norm = self.bn2(x)
		# apply tanh since action must be in range [-1, 1]
		return F.tanh(self.fc3(x_norm))



class Critic(nn.Module):
	""" Critic Model """
	def __init__(self, state_size, action_size, fc1_units, fc2_units, seed):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.bn1 = nn.BatchNorm1d(fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.bn2 = nn.BatchNorm1d(fc2_units)
		self.seed = torch.manual_seed(seed)

	def forward(self, state)
		x = F.relu(self.fc1(state))
		x_norm = self.bn1(x)
		x = F.relu(self.fc2(torch.cat((x_norm, action), dim=1)))
		return self.fc3(x)


















