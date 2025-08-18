from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from itertools import chain
import sys
env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
class D():
    def __init__(self, N=10000):
        self.buffer = collections.deque(maxlen=N)
    def store(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        minibatch = random.sample(self.buffer, n)
        phi_js, a_js, r_js, phi_jp1s, dms = [], [], [], [], []
        for transition in minibatch:
            phi_j, a_j, r_j, phi_jp1, dm = transition
            phi_js.append(phi_j)
            a_js.append(a_j)
            r_js.append(r_j)
            phi_jp1s.append(phi_jp1)
            dms.append(dm)
        return torch.concat(phi_js).to(device), torch.concat(a_js).to(device), \
                torch.concat(r_js).to(device), torch.concat(phi_jp1s).to(device), \
                torch.concat(dms).to(device)
    def size(self):
        return len(self.buffer)
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        self.conv1 = nn.Conv2d(1, 16, 32, 16)
        self.conv2 = nn.Conv2d(16, 20, 8, 4)
        self.atn = nn.Softmax(dim=1)
        self.fcxx = nn.Linear(80, 80)
        self.fcxy = nn.Linear(80, 80)
        self.fcxz = nn.Linear(80, 80)
        self.fcyx = nn.Linear(80, 80)
        self.fcyy = nn.Linear(80, 80)
        self.fcyz = nn.Linear(80, 80)
        self.fczx = nn.Linear(80, 80)
        self.fczy = nn.Linear(80, 80)
        self.fczz = nn.Linear(80, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 7)
    def forward(self, x, y, z):
        x = x.permute(0, 3, 1, 2)
        x = self.grayscale(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fcxx(x) * self.atn(x) + self.fcxy(x) * self.atn(y) + self.fcxz(x) * self.atn(z) + \
            self.fcyx(y) * self.atn(x) + self.fcyy(y) * self.atn(y) + self.fcyz(y) * self.atn(z) + \
            self.fczx(z) * self.atn(x) + self.fczy(z) * self.atn(y) + self.fczz(z) * self.atn(z) 
        x = self.fc2(x) * self.atn(x)
        x = self.fc3(x)
        return x
class Compressor(nn.Module):
    def __init__(self):
        super(Compressor, self).__init__()
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        self.conv1 = nn.Conv2d(1, 16, 32, 16)
        self.conv2 = nn.Conv2d(16, 20, 8, 4)
        self.fc1 = nn.Linear(80, 80)
        self.atn = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.grayscale(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x) * self.atn(x)
        return x
class Future(nn.Module):
    def __init__(self):
        super(Future, self).__init__()
        self.fc1 = nn.Linear(80, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.atn = nn.Softmax(dim=1)
    def forward(self, x):
        x = x + self.fc1(x) * self.atn(x)
        x = x + self.fc2(x) * self.atn(x)
        x = x + self.fc3(x) * self.atn(x)
        return x
class ActionPredictor(nn.Module):
    def __init__(self):
        super(ActionPredictor, self).__init__()
        self.fc1 = nn.Linear(80*2, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 7)
        self.atn = nn.Softmax(dim=1)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x) * self.atn(x)
        x = self.fc3(x)
        return x
def train(q, d, o, f, c, ap):
    # print('training..')
    s, a, r, s_tp1, dm = d.sample(bs)
    cs, cs_tp1 = c(s), c(s_tp1)
    fcs = f(cs)
    f5fcs = f(f(f(f(f(fcs)))))
    q_p = q(s, fcs, f5fcs)
    q_a = q_p.gather(1, a)
    fcs_tp1 = f(cs_tp1)
    f5fcs_tp1 = f(f(f(f(f(fcs_tp1)))))
    q_t = r + g * q(s_tp1, fcs_tp1, f5fcs_tp1).max(1)[0] * dm
    q_t = q_t.unsqueeze(1)
    l1 = F.smooth_l1_loss(q_a, q_t)
    l2 = F.smooth_l1_loss(fcs, cs_tp1)
    st = torch.concat([cs, cs_tp1], dim=1)
    a_p = ap(st)
    # l3 = F.smooth_l1_loss(a_p, F.one_hot(a.long(), num_classes=7).squeeze(1).float())
    l3 = F.cross_entropy(a_p, F.one_hot(a.long(), num_classes=7).squeeze(1).float())
    l = l1 + l2 + l3
    o.zero_grad()
    l.backward()
    o.step()
    # print('grads:', q.fc3.weight.grad[0][0].item(), f.fc3.weight.grad[0][0].item(), c.fc1.weight.grad[0][0].item(), ap.fc3.weight.grad[0][0].item())
    if plot:
        l1s.append(l1.detach())
        l2s.append(l2.detach())
        l3s.append(l3.detach())
def n_p(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def preprocess(state):
    s = torch.from_numpy(state.copy()).float().unsqueeze(0).to(device)
    s /= 255
    s -= 0.5
    s *= 2
    # s += torch.randn_like(s)
    return s
if __name__ == '__main__':
    device = torch.device('cpu')
    q = Q().to(device)
    f = Future().to(device)
    c = Compressor().to(device)
    ap = ActionPredictor().to(device)
    print('q:', n_p(q))
    print('f:', n_p(f))
    print('c:', n_p(c))
    print('ap:', n_p(ap))
    plot = False
    l1s, l2s, l3s, curs = [], [], [], []
    if len(sys.argv)>1 and 'new' in sys.argv[1]:
        pass
    else:
        print('loading..')
        q.load_state_dict(torch.load('mario_q.pth'))
        f.load_state_dict(torch.load('mario_f.pth'))
        c.load_state_dict(torch.load('mario_c.pth'))
        ap.load_state_dict(torch.load('mario_ap.pth'))
    bs = 4
    d = D(int(1e4))
    o = torch.optim.AdamW(chain(q.parameters(), f.parameters(), c.parameters(), ap.parameters()), lr=0.01)
    g = 0.98
    done = True
    for step in range(1, int(1e10)):
        if done:
            state = env.reset()
            state = preprocess(state)
        future = f(c(state))
        future5_future = f(f(f(f(f(future)))))
        action = q(state, future, future5_future).argmax(dim=1)
        state_tp1, reward, done, info = env.step(action.item())
        state_tp1 = preprocess(state_tp1)
        with torch.no_grad():
            curiosity = F.mse_loss(future, c(state_tp1))
        if done:
            reward = torch.tensor(0.0)
        else:
            reward = curiosity
        # print('reward:', reward.item())
        done_mask = torch.tensor([0.0]) if done else torch.tensor([1.0])
        transition = (state, action.unsqueeze(0).long(), reward.unsqueeze(0).float(), state_tp1, done_mask)
        d.store(transition)
        if d.size() >= bs:
            train(q, d, o, f, c, ap)
        env.render()
        if step % 10000 == 0:
            print('saving..')
            torch.save(q.state_dict(), 'mario_q.pth')
            torch.save(f.state_dict(), 'mario_f.pth')
            torch.save(c.state_dict(), 'mario_c.pth')
            torch.save(ap.state_dict(), 'mario_ap.pth')
        if plot:
            curs.append(curiosity)
    env.close()
    if plot:    
        import matplotlib.pyplot as plt 
        plt.plot(l1s, label='policy loss')
        plt.plot(l2s, label='dynamics loss')
        plt.plot(l3s, label='inverse dynamics loss')
        plt.plot(curs, label='curiosities/rewards')
        plt.yscale('log')
        plt.legend()
        plt.savefig('mario.png')