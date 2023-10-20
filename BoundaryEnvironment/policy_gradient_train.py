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
from save_load_model import save_model
from save_load_model import load_model

#============================================================
GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 3
HORIZON = 30
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
    parser.add_argument('-o', '--o_file', help='Output file path')

    # parser.add_argument('-o', '--output', help='Output file path')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    # Parse the command-line arguments
    args = parser.parse_args()
    i_model_path = args.i_file
    o_model_path = args.o_file


    writer = SummaryWriter(comment="-MingiGrid Maze with boundary") # initialize tensorboard writer
    
    env = SimpleEnv(   # initialize the environment
        width=ENV_WIDTH,
        height=ENV_HEIGHT
    )

    env.reset()   # initialize the environment

    state = get_one_hot_encoded_state(env)

    net = PolicyGradientNet(
        input_size=len(state),
        output_size=ACTION_SPACE_SIZE
    )

    if(i_model_path !=None):
        net.load_state_dict(load_model(model_path=i_model_path))


    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_returns = [],[],[]
    episode_rewards = []
    batch_count = 0

    TOTAL_N_EPISODES = int(N_EPOCHS*EPOCH_SIZE)
    for step_idx in range(TOTAL_N_EPISODES):
        env.reset()
        state = get_one_hot_encoded_state(env)
        total_reward = 0
        horizon_counter = 0

        while horizon_counter < HORIZON:
            # Use the policy network to select an action
            horizon_counter+=1
            state_v = torch.FloatTensor(state)  # convert into Float Tensor
            logits_v = net(state) # pass the state into net
            probs_v = F.softmax(logits_v, dim=0)
            action = np.random.choice(len(probs_v), p = probs_v.data.numpy()) # This causes the action to be chosen according to the probability distribution produced from the neural net.

            _, reward, done, _1, _23 = env.step(action)

            next_state = get_one_hot_encoded_state(env)
            total_reward+=reward

            batch_states.append(state)
            batch_actions.append(action)
            batch_returns.append(reward)


            if done:
                returns_from_episode = calculate_discounted_returns(episode_rewards)
                total_rewards.append(returns_from_episode[0])
                batch_returns.extend(returns_from_episode)
                episode_rewards.clear()
                batch_episodes+=1

                if batch_episodes>= EPISODES_TO_TRAIN:
                    batch_count+=1
                    print('batch_finished', batch_count)
                    optimizer.zero_grad()
                    states_v = torch.FloatTensor(batch_states)
                    batch_actions_t = torch.LongTensor(batch_actions)
                    batch_returns_v = torch.FloatTensor(batch_returns)

                    logits_v = net(states_v)
                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    log_prob_actions_v = batch_returns_v * log_prob_v[range(len(batch_states)), batch_actions_t]
                    loss_v = -log_prob_actions_v.mean()
                    writer.add_scalar("loss", loss_v.item(), step_idx)

                    loss_v.backward()
                    optimizer.step()

                    batch_episodes = 0
                    batch_states.clear()
                    batch_actions.clear()
                    batch_returns.clear()

                    save_model(
                        net,
                        o_model_path
                    )


            if done:
                break

            state = next_state

        # handle new rewards
        new_rewards = [total_reward]
        done_episodes += 1
        # total_rewards.append(total_reward)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        # print("%d: reward: %6.5f, mean_100: %6.2f, episodes: %d" % (
        #     step_idx, total_reward, mean_rewards, done_episodes))
        if(len(total_rewards)>0):
            writer.add_scalar("discounted_return", total_rewards[-1], step_idx)
        else:
            writer.add_scalar("discounted_return", 0, step_idx)
        writer.add_scalar("mean_discounted_return_100", mean_rewards, step_idx)
        # writer.add_scalar("episodes", done_episodes, step_idx)
        if mean_rewards > 195:
            print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
            break

    writer.close()







