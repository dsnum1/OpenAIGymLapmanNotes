import gymnasium as gym
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Function to save a PyTorch model to a file
def save_model(model, filename):
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Combine the current directory with the provided filename
    filepath = os.path.join(current_directory, filename)
    
    # Save the model to the specified file
    torch.save(model.state_dict(), filepath)
    
    print(f"Model saved to {filepath}")



def ohc_state(env):
    row, col, _ = 5,5,23
    state = []
    agent_x = env.agent_pos[0]
    agent_y = env.agent_pos[1]
    for i in range(row):
        for j in range(col):
            if(i!=0 and i!=row-1 and j!=0 and j!=col-1):
                if(agent_x == i and agent_y == j):
                    state.append(1)
                else:
                    state.append(0)
    return state    



class PolicyGradientNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyGradientNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )


    def forward(self,x):
        return self.net(torch.FloatTensor(x))
    





# Function to load a PyTorch model from a file
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")



def test_agent(model_path):
    env_id = 'MiniGrid-Empty-5x5-v0'
    env = gym.make(
                env_id,
                tile_size=32,
                render_mode="human",
                agent_pov=False,
                agent_view_size=3,
                screen_size=640,
            )
    obs = env.reset()

    net = PolicyGradientNet(len(ohc_state(env)), output_size=3)
    print(net)

    # Load the trained model weights
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode

    total_reward = 0.0
    num_episodes = 10  # Number of episodes to run for testing

    for _ in range(num_episodes):
        env.reset()
        state = ohc_state(env)
        episode_reward = 0.0

        while True:
            # Use the policy network to select an action
            with torch.no_grad():
                state_v = torch.FloatTensor(state)
                action_probs = F.softmax(net(state_v), dim=0)
                action = torch.multinomial(action_probs, num_samples=1).item()

            _, reward, done, _1, _23 = env.step(action)
            episode_reward += reward

            if done:
                break
            next_state = ohc_state(env)
            state = next_state

        total_reward += episode_reward
        
    average_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")







test_agent('saved_model_to_show')




