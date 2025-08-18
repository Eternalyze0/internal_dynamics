# Internal Dynamics

A neural network with an internal dynamics predictor. The agent first predicts all possible 1-step ahead futures based on all possible actions, then makes a decision based on these futures. The futures network is also trained for consistency with the actual future. The future network loss is added to the reward (curiosity). Baseline DQN code is from https://github.com/seungeunrho/minimalRL. Most relevant paper is https://arxiv.org/abs/1705.05363, which uses an external dynamics predictor. 

So far, it can beat the first level (1-1) of Super Mario Bros:

![mario_win_1-1_clip](https://github.com/user-attachments/assets/11fae889-bf62-4bd9-ab42-5351b9cba6b0)

But the second level (1-2) is much more challenging:

![mario_1-2_clip](https://github.com/user-attachments/assets/1958ad94-09ad-4094-9707-a65b680b318b)

Curiosity spikes during movement:

![mario_graph](https://github.com/user-attachments/assets/b3eaf828-4a04-4047-92ef-fa656602e2f0)



```py
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
```
