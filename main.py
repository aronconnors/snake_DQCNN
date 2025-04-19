#import gym
import numpy as np
from PIL import Image
import torch
from wrapper import *
import os
from model import AtariNet
from agent import Agent
import settings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

environment = DQNenv(device=device)

model = AtariNet(nb_actions=4)

model.to(device)

#model.load_the_model(REQUIRES ARGUMENTS)

agent = Agent(model=model, device=device, epsilon=1.0, min_epsilon=0.1, nb_warmup=5000, nb_actions=4, learning_rate=settings.TRAIN_LR, memory_capacity=1000000, batch_size=64)

agent.train(env=environment, epochs=200000)
