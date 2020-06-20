

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 20       # how often to update the network

LR_ACTOR  = 1e-3		# learning rate of the actor 
LR_CRITIC = 1e-3		# learning rate of the critic 
WEIGHT_DECAY = 0.0		# L2 weight decay
N_LEARN_UPDATE = 10		# number of learning updates
SEED = 42

FC1_UNITS = 256			# Number of nodes in first hidden layer of Actor and Critic
FC2_UNITS = 256			# Number of nodes in second hidden layer of Actor and Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")