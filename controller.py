import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from model import Net

class Controller(nn.Module):

    def __init__(self, num_actions=10, hidden_size=64):
        super(Controller, self).__init__()

        self.cell = nn.GRUCell(
            input_size=num_actions,
            hidden_size=hidden_size
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_actions
        )

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.epsilon = 0.8
        self.gamma = 1.0
        self.max_depth = 6
        self.clip_norm = 0
        self.log_probs = []
        self.actions = []
        self.reward = None

        self.index_to_action = {
            0: 1,
            1: 2,
            2: 4,
            3: 8,
            4: 16,
            5: 'Sigmoid',
            6: 'Tanh',
            7: 'ReLU',
            8: 'LeakyReLU',
            9: 'EOS'
        }

        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-2)


    def forward(self, x, h):
        x = x.unsqueeze(dim=0)
        h = h.unsqueeze(dim=0)

        h = self.cell(x, h)
        x = self.fc(h)

        x = x.squeeze(dim=0)
        h = h.squeeze(dim=0)

        return x, h

    
    def step(self, state):
        logits, new_state = self(torch.zeros(self.num_actions), state)
        
        idx = torch.distributions.Categorical(logits=logits).sample()
        log_probs = logits-torch.logsumexp(logits, dim=0)

        action = self.index_to_action[int(idx)]
        self.actions.append(action)

        terminate = action == 'EOS' or len(self.actions) == self.max_depth

        return log_probs[idx], new_state, terminate


    def generate_rollout(self, iter_train, iter_dev):
        state = torch.zeros(self.hidden_size)
        terminated = False

        while not terminated:
            log_probs, state, terminated = self.step(state)
            self.log_probs.append(log_probs)

        print('\nGenerated network:')
        print(self.actions)

        net = Net(self.actions)
        accuracy = net.fit(iter_train, iter_dev)
        self.reward = accuracy - 1

        return self.reward

    
    def optimize(self):
        G = torch.ones(1) * self.reward
        loss = 0

        for i in reversed(range(len(self.log_probs))):
            G = self.gamma * G
            loss = loss - (self.log_probs[i]*Variable(G))
        
        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        self.optimizer.step()

        self.log_probs = []
        self.actions = []
        self.reward = None

        return float(loss.data.numpy())
