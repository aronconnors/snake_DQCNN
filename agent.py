import random
import torch
import copy
import torch.optim as optim
import torch.nn.functional as F
from plot import livePlot
import numpy as np
import time
import settings

class ReplayMemory:

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0

    def insert(self, transition):
        transition = [item.to('cpu') for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10
    
    def __len__(self):
        return len(self.memory)
        

class Agent:
    def __init__(self, model, device, epsilon, min_epsilon, nb_warmup, nb_actions, memory_capacity, batch_size, learning_rate):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        if settings.TARGET_NETWORK:
            self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size
        self.model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        print(f"Starting epsilon is {self.epsilon}")
        print(f"Epsilon decay is {self.epsilon_decay}")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))

        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)
        
    def train(self, env, epochs):
        stats = {"Returns": [], "AvgReturns": [], "Scores": [], "HighScore": [], "Loss": [], "EpsilonCheckpoint": []}
        plotter = livePlot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, score = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    if settings.TARGET_NETWORK:
                        next_qsa_b = self.target_model(next_state_b)
                    else:
                        next_qsa_b = self.model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return +=reward.item()
            
            stats["Returns"].append(ep_return)
            stats["Loss"].append(loss)
            stats["Scores"].append(score)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay
            
            if epoch % 10 == 0:
                self.model.save_the_model(f'models/{settings.MODEL_NAME}')
                print(" ")

                average_returns = np.mean(stats["Returns"][-100:])
                average_scores = np.mean(stats["Scores"][-100:])
                high_score = np.max(stats["Scores"][:])

                stats["AvgReturns"].append(average_returns)
                stats["AvgScores"].append(average_scores)
                stats["HighScore"].append(high_score)
                stats["EpsilonCheckpoint"].append(self.epsilon)

                if (len(stats["Returns"])) > 100:
                    print(f"Epoch: {epoch} - Average Return (prev 100): {np.mean(stats['Returns'][-100:])} - Average Score (prev 100): {np.max(stats['Scores'][-100:])} - High Score: {np.mean(stats['Scores'][:])} - Epsilon: {self.epsilon}")
                else:
                    print(f"Epoch: {epoch} - Episode Return: {np.mean(stats['Returns'][-1:])} - Episode Score: {np.mean(stats['Scores'][-1:])} - High Score: {np.mean(stats['Scores'][:])} - Epsilon: {self.epsilon}")

            if epoch % 100 == 0:
                if settings.TARGET_NETWORK:
                    self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/{settings.MODEL_NAME}_{epoch}.pt")

        return stats
    
    def test(self, env):
        for epoch in range(1,3):
            state = env.reset()

            done = False
            
            while not done:
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, score = env.step(action)
            print(score)

                


