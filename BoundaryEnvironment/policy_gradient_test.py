import gymnasium as gym
import numpy as np
import argparse
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from boundary_env import SimpleEnv
from PolicyGradientNet import PolicyGradientNet
from save_load_model import load_model

#============================================================
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPISODES_TO_TRAIN = 3
HORIZON = 50
ENV_WIDTH=9
ENV_HEIGHT=9
ACTION_SPACE_SIZE=3
EPOCH_SIZE = 1000
N_EPOCHS= 100
#============================================================

def calculate_discounted_returns(rewards):
    discounted_returns = []
    running_return = 0.0
    for r in reversed(rewards):
        running_return*=GAMMA
        running_return+= r
        discounted_returns.append(running_return)
    
    return list(reversed(discounted_returns))


def get_one_hot_encoded_state(env):
    from pprint import pprint
    agent_x = env.agent_pos[0]
    agent_y = env.agent_pos[1]
    state = env.get_state(agent_x, agent_y)
    # print("(", agent_x, agent_y,")" ,state.index(1))
    # state = state.reshape(-1)

    # print(env.observation_space.spaces['image'])

    # print(env.grid.encode())
    # row, col, _ = 5,5,23
    # state = []
    # for i in range(row):
    #     for j in range(col):
    #         if(i!=0 and i!=row-1 and j!=0 and j!=col-1):
    #             if(agent_x == i and agent_y == j):
    #                 state.append(1)
    #             else:
    #                 state.append(0)
    return state    

    

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Execute training for a policy gradient function here')
    # Define command-line arguments
    parser.add_argument('-f', '--i_file', help='Input file path')
    # parser.add_argument('-o', '--o_file', help='Output file path')

    # parser.add_argument('-o', '--output', help='Output file path')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    # Parse the command-line arguments
    args = parser.parse_args()
    i_model_path = args.i_file

    env = SimpleEnv(render_mode="human", width=9, height=9)
    env.reset()
    net = PolicyGradientNet(len(get_one_hot_encoded_state(env)), output_size=3)
    print(net)

    # Load the trained model weights
    net.load_state_dict(torch.load(i_model_path))
    net.eval()  # Set the model to evaluation mode



    total_reward = 0.0
    num_episodes = 10  # Number of episodes to run for testing

    for _ in range(num_episodes):
        state = get_one_hot_encoded_state(env)
        episode_reward = 0.0
        env.reset()

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
            next_state = get_one_hot_encoded_state(env)
            state = next_state

        total_reward += episode_reward

    average_reward = total_reward / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {average_reward:.2f}")










