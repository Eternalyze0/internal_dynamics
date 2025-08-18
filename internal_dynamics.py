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
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
class D():
    def __init__(self, N=10000):
        self.buffer = collections.deque(maxlen=N)
    def store(self, transition):
        self.buffer.append(transition)
    def sample(self, n):
        minibatch = random.sample(self.buffer, n)
        phi_js, a_js, r_js, phi_jp1s = [], [], [], []
        for transition in minibatch:
            phi_j, a_j, r_j, phi_jp1 = transition
            phi_js.append(phi_j)
            a_js.append(a_j)
            r_js.append(r_j)
            phi_jp1s.append(phi_jp1)
        return torch.concat(phi_js).to(device), torch.concat(a_js).to(device), \
                torch.concat(r_js).to(device), torch.concat(phi_jp1s).to(device)
    def size(self):
        return len(self.buffer)
class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
        self.conv1 = nn.Conv2d(1, 16, 32, 16)
        self.conv2 = nn.Conv2d(16, 20, 8, 4)
        self.fc1 = nn.Linear(80, 80)
        self.atn = nn.Softmax(dim=1)
        self.fcf = nn.Linear(80, 80)
        self.fcff= nn.Linear(80, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 7)
    def forward(self, x, y, z):
        x = x.permute(0, 3, 1, 2)
        x = self.grayscale(x)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = self.fc1(x) * self.atn(x) + self.fcf(y) * self.atn(y) + self.fcff(z) * self.atn(z) 
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
        x = x + F.leaky_relu(self.fc1(x) * self.atn(x))
        x = x + F.leaky_relu(self.fc2(x) * self.atn(x))
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
    s, a, r, s_tp1 = d.sample(bs)
    cs, cs_tp1 = c(s), c(s_tp1)
    fcs = f(cs)
    ffcs = f(fcs)
    q_p = q(s, fcs, ffcs)
    q_a = q_p.gather(1, a)
    fcs_tp1 = f(cs_tp1)
    ffcs_tp1 = f(fcs_tp1)
    q_t = r + g * q(s_tp1, fcs_tp1, ffcs_tp1).max(1)[0]
    q_t = q_t.unsqueeze(1)
    l1 = F.smooth_l1_loss(q_a, q_t)
    l2 = F.mse_loss(fcs, cs_tp1)
    st = torch.concat([cs, cs_tp1], dim=1)
    a_p = ap(st)
    l3 = F.smooth_l1_loss(a_p, F.one_hot(a.long(), num_classes=7).squeeze(1).float())
    l = l1 + l2 + l3
    o.zero_grad()
    l.backward()
    o.step()
    print('grads:', q.fc3.weight.grad[0][0].item(), f.fc3.weight.grad[0][0].item(), c.fc1.weight.grad[0][0].item(), ap.fc3.weight.grad[0][0].item())
def n_p(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    o = torch.optim.AdamW(chain(q.parameters(), f.parameters(), c.parameters(), ap.parameters()), lr=0.0005)
    g = 0.98
    done = True
    for step in range(1, int(1e10)):
        if done:
            state = env.reset()
            state = torch.from_numpy(state.copy()).float().unsqueeze(0).to(device)
        future = f(c(state))
        future_future = f(future)
        action = q(state, future, future_future).argmax(dim=1)
        state_tp1, reward, done, info = env.step(action.item())
        state_tp1 = torch.from_numpy(state_tp1.copy()).float().unsqueeze(0).to(device)
        with torch.no_grad():
            curiosity = F.mse_loss(future, c(state_tp1))
        reward = curiosity
        print(reward.item())
        transition = (state, action.unsqueeze(0).long(), reward.unsqueeze(0).float(), state_tp1)
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
    env.close()