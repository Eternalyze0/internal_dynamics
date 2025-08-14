import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
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

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fcis = nn.Linear(4, 128) # fully connected input state
        self.fcof = nn.Linear(128, 4*2) # fully connected output futures
        self.fcif = nn.Linear(4*2, 128) # fully connected input futures
        self.fcoa = nn.Linear(128, 2) # fully connected output actions

    def future(self, s):
        if len(s.shape)==1:
            s = s.unsqueeze(0)
        x = F.relu(self.fcis(s))
        x = self.fcof(x)
        x = x.reshape(x.shape[0], 2, 4)
        return x

    def forward(self, f):
        f = f.reshape(f.shape[0], 8)
        x = F.relu(self.fcif(f))
        x = self.fcoa(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(q.future(s))
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(q_target.future(s_prime)).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_future(q, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        s_prime_prediction = q.future(s).gather(1, a.unsqueeze(-1).expand(-1, -1, 4))
        loss = F.mse_loss(s_prime_prediction, s_prime.reshape(batch_size, 1, 4))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        # epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        epsilon = 0.0
        s, _ = env.reset()
        done = False

        while not done:
            f = q.future(torch.from_numpy(s).float())
            a = q.sample_action(f, epsilon)      
            s_prime, r, done, truncated, info = env.step(a)
            r += F.mse_loss(f.gather(1, torch.tensor([a], dtype=torch.long).view(1, 1).unsqueeze(-1).expand(-1, -1, 4)), 
                torch.tensor(s_prime))
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)
            train_future(q, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()