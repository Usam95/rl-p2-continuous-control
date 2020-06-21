
# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.
In this project an agent is trained to move its double-jointed arm to targets location.

**Reward**: A reward of +0.1 is provided for each step that the agent's hand is in the goal location.

**Goal**: The goal of your agent is to maintain its position at the target location for as many time steps as possible.

**Observation Space**: The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints.

**Action Space**: Every entry in the action vector should be a number between -1 and 1.

In order to consider the environment has been solved, the agent must get an average score of +30 over 100 consecutive episodes.
![Trained Agent][./images/joint-arm.png]


### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

## Getting Started

1. Configure a new conda virtual environment for Python 3.6 with the needed requirements as described in this [Udacity](https://github.com/udacity/deep-reinforcement-learning#dependencies) repository. 
2. Install Unity Agents using: `pip install unityagents`
2. The environment provided with this repository is for Mac OS. If you have another OS than please use one of the following links to donwload the environment for your OS replacing the "Banana.app" in this repository.  
Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

### Instructions

Follow the instructions in `Solution.ipynb` to get started with training your own agent! 

