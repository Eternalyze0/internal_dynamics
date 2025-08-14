import collections
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import ale_py
from itertools import chain

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 32
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
        #        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
        #        torch.tensor(done_mask_lst)
    
        return torch.stack(s_lst, dim=0), torch.tensor(a_lst), \
                torch.tensor(r_lst), torch.stack(s_prime_lst, dim=0), \
                torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)

class Map(nn.Module):
    def __init__(self):
        super(Map, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc = nn.Linear(1344, 4)
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=1)

    def forward(self, x):
        if len(x.shape)==2:
            x = x.unsqueeze(0)
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        x = self.grayscale(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fcis = nn.Linear(4, 128) # fully connected input state
        self.fcof = nn.Linear(128, 4*7) # fully connected output futures
        self.fcif = nn.Linear(4*7, 128) # fully connected input futures
        self.fcoa = nn.Linear(128, 7) # fully connected output actions

    def future(self, s): # (240, 256, 3)
        if len(s.shape)==1:
            s = s.unsqueeze(0)
        x = F.relu(self.fcis(s))
        x = self.fcof(x)
        x = x.reshape(x.shape[0], 7, 4)
        return x

    def forward(self, f):
        f = f.reshape(f.shape[0], 28)
        x = F.relu(self.fcif(f))
        x = self.fcoa(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer, m):
    for i in range(1):
        ms,a,r,ms_prime,done_mask = memory.sample(batch_size)

        q_out = q(q.future(ms))
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(q_target.future(ms_prime)).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_future(q, memory, optimizer, m):
    for i in range(1):
        ms,a,r,ms_prime,done_mask = memory.sample(batch_size)

        s_prime_prediction = q.future(ms).gather(1, a.unsqueeze(-1).expand(-1, -1, 4))
        loss = F.mse_loss(s_prime_prediction, ms_prime)

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()

def main():
    m = Map()
    q = Qnet()
    if len(sys.argv)>1 and 'new' in sys.argv[1]:
        pass
    else:
        m.load_state_dict(torch.load('mario_m.pth'))
        q.load_state_dict(torch.load('mario_q.pth'))
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1 #20
    score = 0.0  
    optimizer = optim.Adam(chain(q.parameters(), m.parameters()), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            ms = m(torch.from_numpy(s.copy()).float()).unsqueeze(0)
            del s
            f = q.future(ms)
            a = q.sample_action(f, epsilon)      
            s_prime, r, done, info = env.step(a)
            env.render()
            ms_prime = m(torch.tensor(s_prime.copy(), dtype=torch.float)).unsqueeze(0)
            c = F.mse_loss(f.gather(1, torch.tensor([a], dtype=torch.long).view(1, 1).unsqueeze(-1).expand(-1, -1, 4)), 
                ms_prime)
            print(c)
            r += c
            done_mask = 0.0 if done else 1.0
            # print(ms.shape, ms_prime.shape)
            memory.put((ms,a,r/100.0,ms_prime, done_mask))
            s = s_prime
            r -= c
            score += r.item()
            if done:
                break
            
        if memory.size()>batch_size:
            train(q, q_target, memory, optimizer, m)
            train_future(q, memory, optimizer, m)

        if n_epi%print_interval==0 and n_epi!=0:
            torch.save(q.state_dict(), 'mario_q.pth')
            torch.save(m.state_dict(), 'mario_m.pth')
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()