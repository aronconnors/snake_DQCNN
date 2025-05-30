import torch
import torch.nn as nn
import os
import settings

class AtariNet(nn.Module):
    def __init__(self, nb_actions=4):
        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2)

        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_actions)

        if settings.MODEL_ARCHITECTURE == 'dueling':
            self.state_value1 = nn.Linear(3136, 1024)
            self.state_value2 = nn.Linear(1024, 1024)
            self.state_value3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)

        if settings.MODEL_ARCHITECTURE == 'dueling':
            state_value = self.relu(self.state_value1(x))
            state_value = self.dropout(state_value)
            state_value = self.relu(self.state_value2(state_value))
            state_value = self.dropout(state_value)
            state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)
        
        if settings.MODEL_ARCHITECTURE == 'dueling':
            output = state_value + (action_value - action_value.mean())
            return output

        if settings.MODEL_ARCHITECTURE == 'base':
            return action_value

    def save_the_model(self, weights_filename):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename=f'models/savedModels/{settings.LOAD_MODEL_NAME}'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print(f"Successfully loaded weights file {weights_filename}")
        except FileNotFoundError:
            print(f"Error: Weights file not found at {weights_filename}")
        except RuntimeError as e:
            print(f"Error loading weights from {weights_filename}: {e}")
        except Exception as e:  # Catch any other potential errors
            print(f"An unexpected error occurred while loading {weights_filename}: {e}")
