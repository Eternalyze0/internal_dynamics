# Internal Dynamics

A neural network with an internal dynamics predictor that robustly beats cartpole without epsilon-greedy. The agent first predicts all possible 1-step ahead futures based on all possible actions, then makes a decision based on these futures. The futures network is also trained for consistency with the actual future. The future network loss is added to the reward (curiosity).

```py
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fcis = nn.Linear(4, 128)
        self.fcof = nn.Linear(128, 4*2)
        self.fcif = nn.Linear(4*2, 128)
        self.fcoa = nn.Linear(128, 2)

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
```
