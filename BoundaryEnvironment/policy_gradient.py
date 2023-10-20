import gymnasium as gym
import numpy as np
import os
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


HORIZON = 50
from boundary_env import SimpleEnv


# Function to save a PyTorch model to a file
def save_model(model, filename):
    # Get the current working directory
    current_directory = os.getcwd()
    
    # Combine the current directory with the provided filename
    filepath = os.path.join(current_directory, filename)
    
    # Save the model to the specified file
    torch.save(model.state_dict(), filepath)
    
    print(f"Model saved to {filepath}")






# Function to load a PyTorch model from a file
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")


GAMMA = 0.99
LEARNING_RATE = 0.0001
EPISODES_TO_TRAIN = 3


def ohc_state(env):
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
    

def calculate_discounted_returns(rewards):
    discounted_returns = []
    running_return = 0.0
    for r in reversed(rewards):
        running_return*=GAMMA
        running_return+= r
        discounted_returns.append(running_return)
    
    return list(reversed(discounted_returns))
    

def train(model_path):
    # env = gym.make("CartPole-v0")
    # env_id = 'MiniGrid-Empty-5x5-v0'
    # env = gym.make(env_id)
    writer = SummaryWriter(comment="-MingiGrid Maze with boundary")

    # env = gym.make(
    #             env_id,
    #             tile_size=32,
    #             # render_mode="human",
    #             agent_pov=False,
    #             agent_view_size=3,
    #             screen_size=640,
    #         )

    env = SimpleEnv(
        # render_mode="human", 
        width=9, height=9)
    # env = ImgObsWrapper(env)  # Wrap the environment to provide image observations

    obs = env.reset()

    # print(ohc_state(env))
    # # writer = SummaryWriter(comment="-cartpole-reinforce")
    
    net = PolicyGradientNet(len(ohc_state(env)), output_size=3)
    if model_path!=None:
        print('model_loaded')
        # net.load_state_dict(torch.load(model_path))
    print(net)

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)


    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_returns = [],[],[]
    episode_rewards = []
    batch_count = 0




    for step_idx in range(2000000):
        state = env.reset()
        state = ohc_state(env)


        unchecked_batch_states = 0
        unchecked_batch_states = []
        unchecked_batch_actions = []
        unchecked_episode_rewards = []
        unchecked_batch_returns = []

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

            next_state = ohc_state(env)
            total_reward+=reward

            unchecked_batch_states.append(state)
            unchecked_batch_actions.append(action)
            unchecked_episode_rewards.append(reward)

            if done:
                batch_states+=unchecked_batch_states
                batch_actions+=unchecked_batch_actions
                episode_rewards+=unchecked_episode_rewards

                returns_from_episode = calculate_discounted_returns(episode_rewards)
                total_rewards.append(returns_from_episode[0])
                batch_returns.extend(returns_from_episode)
                episode_rewards.clear()
                batch_episodes+=1

                if batch_episodes>= EPISODES_TO_TRAIN:
                    print('batch_length', len(batch_states))
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
                        model_path
                    )


            
            if done:
                break

            state = next_state



        unchecked_batch_states.clear()
        unchecked_batch_actions.clear()
        unchecked_episode_rewards.clear()

        # handle new rewards
        new_rewards = [total_reward]
        done_episodes += 1
        # total_rewards.append(total_reward)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        print("%d: reward: %6.5f, mean_100: %6.2f, episodes: %d" % (
            step_idx, total_reward, mean_rewards, done_episodes))
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



    save_model(
            net,
            model_path
        )





def test_agent(model_path):
    # env_id = 'MiniGrid-Empty-5x5-v0'
    # env = gym.make(
    #             env_id,
    #             tile_size=32,
    #             render_mode="human",
    #             agent_pov=False,
    #             agent_view_size=3,
    #             screen_size=640,
    #         )


    env = SimpleEnv(render_mode="human", width=9, height=9)
    env.reset()
    net = PolicyGradientNet(len(ohc_state(env)), output_size=3)
    print(net)

    # Load the trained model weights
    net.load_state_dict(torch.load(model_path))
    net.eval()  # Set the model to evaluation mode

    total_reward = 0.0
    num_episodes = 10  # Number of episodes to run for testing






# train(model_path="b_3_h_50_E_2M_LR_0_0001")
# test_agent('b_3_h_50_E_2M_LR_0_0001')
# main()

# train(model_path='trying_something')
test_agent('trying_something')


# increase the learning rate from -3 to -2
#checkout hindisght experience replay
# intialize SEED
# tc.random(SEED)
# np.random(SEED)
# random.seed(SEED)





