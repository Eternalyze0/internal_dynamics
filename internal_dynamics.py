import collections
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import time

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
# env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = gym_super_mario_bros.make('SuperMarioBros-1-2-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = int(1e4)
batch_size    = 256
plot = False

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

class Attention(nn.Module):
    def __init__(self, n_embd=7):
        super(Attention, self).__init__()
        self.n_embd = n_embd
        self.bias = True
        self.flash = True
        self.n_head = n_embd
        self.dropout = 0.1
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.n_head = self.n_head
        self.n_embd = self.n_embd
        self.dropout = self.dropout  

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y  

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_embd = 7
        self.bias = True
        self.c_fc    = nn.Linear(self.n_embd, 4 * self.n_embd, bias=self.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * self.n_embd, self.n_embd, bias=self.bias)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Block(nn.Module):

    def __init__(self, n_embd=7):
        super().__init__()
        # self.n_embd = n_embd
        self.bias = True
        # self.ln_1 = LayerNorm(self.n_embd, bias=self.bias)
        self.attn = Attention(n_embd=n_embd)
        # self.ln_2 = LayerNorm(self.n_embd, bias=self.bias)
        # self.mlp = MLP()

    def forward(self, x):
        # x = x + self.attn(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))
        x = x + self.attn(x)
        # x = x + self.mlp(x)
        return x

class Organism(nn.Module):
    # def __init__(self):
    #     super(Organism, self).__init__()
    #     self.fcif = nn.Linear(4*7, 128) # fully connected input futures
    #     self.fcoa = nn.Linear(128, 7) # fully connected output actions
    #     self.fitness = None
    #     self.n_f = 0
    #     self.time_of_origin = time.time()

    def __init__(self):
        super(Organism, self).__init__()

        self.fitness = None
        self.n_f = 0
        self.time_of_origin = time.time()
        self.n_embd = 7
        self.vocab_size = 7
        self.bias = True
        self.n_layer = 1

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            # drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block() for _ in range(self.n_layer)]),
            # ln_f = LayerNorm(self.n_embd, bias=self.bias),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=self.bias)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        for block in self.transformer.h:
            x = block(x)
        # x = self.transformer.ln_f(x)
        # logits = self.lm_head(x.mean(dim=1))
        # logits = self.lm_head(x[:,-1,:])
        logits = x.mean(dim=1)
        return logits
        # assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.n_embd = 7
        # self.bias = True
        # self.flash = True
        # self.n_head = 7
        # self.dropout = 0.1



        # self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
        # # output projection
        # self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        # # regularization
        # self.attn_dropout = nn.Dropout(self.dropout)
        # self.resid_dropout = nn.Dropout(self.dropout)
        # self.n_head = self.n_head
        # self.n_embd = self.n_embd
        # self.dropout = self.dropout

    # def forward(self, f):
    #     f = f.reshape(f.shape[0], 28)
    #     x = F.relu(self.fcif(f))
    #     x = self.fcoa(x)
    #     return x

    # def forward(self, x): # torch.Size([1, 7, 4])

    #     x = x.permute(0, 2, 1)

    #     B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    #     # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    #     q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    #     k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    #     v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    #     # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    #     if self.flash:
    #         # efficient attention using Flash Attention CUDA kernels
    #         y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
    #     else:
    #         # manual implementation of attention
    #         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #         att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    #         att = F.softmax(att, dim=-1)
    #         att = self.attn_dropout(att)
    #         y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    #     y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

    #     # output projection
    #     y = self.resid_dropout(self.c_proj(y))
    #     # print(y.shape) # torch.Size([1, 4, 7])
    #     y = y.mean(dim=1)
    #     # print(y.shape) # torch.Size([1, 7])
    #     return y

class Qnet(nn.Module):
    def __init__(self, n_pop=10):
        super(Qnet, self).__init__()
        self.fcis = nn.Linear(4, 128) # fully connected input state
        self.fcof = nn.Linear(128, 4*7) # fully connected output futures
        self.population = nn.ModuleList([Organism() for _ in range(n_pop)])
        # for _ in range(n_pop):
            # self.population.append(Organism())
        self.n_pop = n_pop
        self.evolution_stage = 0

        # self.action_expand = nn.Linear(4, 7*4)

        # self.rif = nn.Linear(4*7, 128)
        # self.rooi = nn.Linear(128, self.n_pop)
        # self.rbrain = nn.Linear(128, 128)
        # self.rdo = nn.Dropout(0.5)

        # self.fblock = Block(n_embd=4)

    def future(self, s): # (240, 256, 3)
        if len(s.shape)==1:
            s = s.unsqueeze(0)
        x = F.relu(self.fcis(s))
        x = self.fcof(x)
        x = x.reshape(x.shape[0], 7, 4)
        return x

    # def future(self, s):
    #     if len(s.shape)==1:
    #         s = s.unsqueeze(0)
    #     if len(s.shape)==4:
    #         s = s.squeeze(1)
    #     # print(s.shape)
    #     # s = s.expand(s.shape[0], 7, s.shape[2])
    #     s = s.flatten(start_dim=1)
    #     s = self.action_expand(s)
    #     # print(s.shape)
    #     s = s.reshape(s.shape[0], 7, 4)
    #     x = self.fblock(s)
    #     # print(x.shape)
    #     # exit()
    #     return x

    def forward(self, f, orsm_idx=0):
        x = self.population[orsm_idx](f)
        return x

    # def route(self, f):
    #     f = f.reshape(f.shape[0], 28)
    #     x = F.relu(self.rif(f))
    #     x = F.relu(self.rbrain(x))
    #     x = self.rdo(x)
    #     x = self.rooi(x)
    #     return x

    def evolve(self, curiosity):
        def mix_parameters(parent1, parent2):
            offspring = Organism()
            
            # Iterate through each parameter in the model
            for (name1, param1), (name2, param2), offspring_param in zip(parent1.named_parameters(), 
                                                                     parent2.named_parameters(), 
                                                                     offspring.parameters()):
                # Ensure we are mixing corresponding parameters
                assert name1 == name2, "Parameter names do not match!"
                
                # Flatten the parameters to 1D for easier manipulation
                param1_flat = param1.data.view(-1)
                param2_flat = param2.data.view(-1)
                offspring_param_flat = offspring_param.data.view(-1)
                
                # Determine the number of parameters to take from each parent and random init
                total_params = param1_flat.size(0)
                one_third = total_params // 3
                
                # Randomly permute the indices
                indices = torch.randperm(total_params)
                
                # Assign 1/3 from parent1, 1/3 from parent2, and 1/3 random
                offspring_param_flat[indices[:one_third]] = param1_flat[indices[:one_third]]
                offspring_param_flat[indices[one_third:2*one_third]] = param2_flat[indices[one_third:2*one_third]]
                
                # Initialize the remaining 1/3 randomly
                # remaining_indices = indices[2*one_third:]
                # if len(remaining_indices) > 0:
                #     # Use Xavier initialization for weights, zeros for biases
                #     if 'weight' in name1:
                #         nn.init.xavier_uniform_(offspring_param_flat[remaining_indices])
                #     elif 'bias' in name1:
                #         nn.init.zeros_(offspring_param_flat[remaining_indices])
                
                # Reshape back to original shape
                offspring_param.data = offspring_param_flat.view(param1.data.shape)
            
            return offspring

        worst_organism = None
        worst_fitness = np.inf
        worst_index = None
        for i, organism in enumerate(self.population):
            if organism.n_f > 1:
                if organism.fitness < worst_fitness:
                    worst_organism = organism
                    worst_fitness = organism.fitness
                    worst_index = i

        best_organism = None
        best_fitness = -np.inf
        best_index = None
        for i, organism in enumerate(self.population):
            if organism.n_f > 1:
                if organism.fitness > best_fitness:
                    best_organism = organism
                    best_fitness = organism.fitness
                    best_index = i

        if worst_organism and best_organism:
            print('population event')
            coin = random.random()
            if coin < 1.0:
                coin = random.random()
                if coin > curiosity/worst_fitness and coin > 0.999:
                    self.population[worst_index] = Organism()
                else:
                    self.population[worst_index] = best_organism

        average_lifespan = 0.0
        for organism in self.population:
            average_lifespan += time.time() - organism.time_of_origin
        average_lifespan /= len(self.population)

        print('best_fitness', best_fitness)
        print('worst_fitness', worst_fitness)
        print('average_lifespan', average_lifespan)

        # first = second = None
        # for obj in self.population:
        #     if first is None or (obj.fitness != None and obj.fitness > first.fitness):
        #         second = first
        #         first = obj
        #     elif second is None or (obj.fitness != None and obj.fitness > second.fitness):
        #         second = obj
        # if self.evolution_stage % 10 == 0:
        #     self.population[np.argmin([x.fitness for x in self.population])] = first
        # if self.evolution_stage % 11 == 0:
        #     self.population[np.argmin([x.fitness for x in self.population])] = Organism()
            # mix_parameters(first, second)
        # best = random.randint(0, self.n_pop-1)
        # second_best = random.randint(0, self.n_pop-1)
        # new_random = random.randint(0, self.n_pop-1)
        # self.population[best] = first
        # if second_best != best:
        #     self.population[second_best] = second
        # if new_random != best and new_random != second_best and self.evolution_stage % 1000 == 0:
        #     self.population[new_random] = Organism()
        self.evolution_stage += 1

      
    def sample_action(self, obs, epsilon, orsm_idx=0):
        out = self.forward(obs, orsm_idx)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,3)
        else : 
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer, m, n_pop):
    for i in range(1):
        ms,a,r,ms_prime,done_mask = memory.sample(batch_size)

        q_out = q(q.future(ms), orsm_idx=random.randint(0, n_pop-1))
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(q_target.future(ms_prime)).max(1)[0].unsqueeze(1)
        # gamma = 0.98 + 0.01 * math.sin(time.time())
        # print('gamma', gamma)
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
        loss.backward()
        optimizer.step()

def train_all(q, q_target, memory, optimizer, m, n_pop, curiosity):
    for i in range(1):
        ms,a,r,ms_prime,done_mask = memory.sample(batch_size)

        s_prime_prediction = q.future(ms.detach()).gather(1, a.unsqueeze(-1).expand(-1, -1, 4))

        q_out = q(q.future(ms.detach()), orsm_idx=random.randint(0, n_pop-1))
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(q_target.future(ms_prime.detach())).max(1)[0].unsqueeze(1)
        # gamma = 0.98 + 0.01 * math.sin(time.time())
        # print('gamma', gamma)
        target = r + gamma * max_q_prime * done_mask
        alpha = 100.0
        loss = F.smooth_l1_loss(q_a, target) + alpha * F.mse_loss(s_prime_prediction, ms_prime.detach())
        # print('loss', loss)
        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
            # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        # q.evolve(curiosity)
        # print(q..weight.grad)
        # for i, organism in enumerate(q.population):
        #     print(chr(97+i) + '.', round(time.time() - organism.time_of_origin), end=' ')
        # print()

def main():
    n_pop=1
    m = Map()
    q = Qnet(n_pop=n_pop)
    if len(sys.argv)>1 and 'new' in sys.argv[1]:
        pass
    else:
        m.load_state_dict(torch.load('mario_m.pth'))
        q.load_state_dict(torch.load('mario_q.pth'))
    q_target = Qnet(n_pop=n_pop)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 1 #20
    score = 0.0  
    optimizer = optim.Adam(chain(q.parameters(), m.parameters()), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        curiosities = []

        while not done:
            ms = m(torch.from_numpy(s.copy()).float()).unsqueeze(0)
            del s
            f = q.future(ms)
            orsm_idx = random.randint(0, n_pop-1)
            # orsm_idx = q.route(f).argmax()
            # print('route', orsm_idx.item())
            # a = q.sample_action(f, epsilon, orsm_idx)
            a = q(f, orsm_idx).argmax().item()
            # print(a.shape)  
            env.render()
            s_prime, r, done, info = env.step(a)
            ms_prime = m(torch.tensor(s_prime.copy(), dtype=torch.float)) #.unsqueeze(0)
            c = F.mse_loss(f.gather(1, torch.tensor([a], dtype=torch.long).view(1, 1).unsqueeze(-1).expand(-1, -1, 4)), 
                ms_prime.unsqueeze(0))
            if q.population[orsm_idx].n_f == 0:
                q.population[orsm_idx].fitness = c.item()
                q.population[orsm_idx].n_f += 1
            else:
                q.population[orsm_idx].fitness = \
                    (q.population[orsm_idx].fitness * q.population[orsm_idx].n_f + c.item()) / (q.population[orsm_idx].n_f+1)
                q.population[orsm_idx].n_f += 1
            # q.evolve()
            # print('curiosity', c)
            r = c
            done_mask = 0.0 if done else 1.0
            # print(ms.shape, ms_prime.shape)
            memory.put((ms,a,r/100.0,ms_prime, done_mask))
            s = s_prime
            # r -= c
            score += r.item()
            if plot:
                curiosities.append(c.item())
                print(curiosities)
                plt.clf()
                plt.ylim(0.0, 1000.0)
                plt.plot(curiosities)
                plt.savefig('mario_plot.png')

            
            if memory.size()>=batch_size:
                # print('training..')
                # train(q, q_target, memory, optimizer, m, n_pop)
                # train_future(q, memory, optimizer, m)
                train_all(q, q_target, memory, optimizer, m, n_pop, c.item())

            if done:
                break

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