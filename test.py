#import gym
import numpy as np
from PIL import Image
import torch
from wrapper import *
import os
from model import AtariNet
from agent import Agent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

environment = DQNenv(device=device, render_mode='human')

model = AtariNet(nb_actions=4)

model.to(device)

'''models = []
for filename in os.listdir('models/savedModels'):
    if filename.endswith('.pt'):
        models.append(filename)

for testModel in models:
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print(testModel)'''

model.load_the_model() #'models/savedModels/'+testModel)

agent = Agent(model=model, device=device, epsilon=0.005, nb_warmup=5000, min_epsilon=0, nb_actions=4, learning_rate=0.00001, memory_capacity=1000000, batch_size=64)

agent.test(env=environment)
