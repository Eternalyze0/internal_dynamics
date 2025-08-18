# Internal Dynamics

A neural network with an internal dynamics predictor. The agent predicts the 1-step and 2-step ahead futures, then makes a decision based on these futures. The futures network is also trained for consistency with the actual future. The future network loss is the reward and motivation of the agent (curiosity). Implementation initially closesly follows https://arxiv.org/abs/1705.05363, https://arxiv.org/abs/1312.5602, and https://github.com/seungeunrho/minimalRL/blob/master/dqn.py .

So far, it can beat the first level (1-1) of Super Mario Bros:

![mario_win_1-1_clip](https://github.com/user-attachments/assets/11fae889-bf62-4bd9-ab42-5351b9cba6b0)

But the second level (1-2) is much more challenging:

![mario_1-2_clip](https://github.com/user-attachments/assets/1958ad94-09ad-4094-9707-a65b680b318b)

Curiosity spikes during movement:

![mario_graph](https://github.com/user-attachments/assets/b3eaf828-4a04-4047-92ef-fa656602e2f0)

## Usage

```
python3.10 interal_dynamics.py
```
