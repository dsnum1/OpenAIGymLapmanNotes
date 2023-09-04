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






# Function to load a PyTorch model from a file
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {filepath}")


GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4


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
    

def calculate_discounted_returns(rewards):
    discounted_returns = []
    running_return = 0.0
    for r in reversed(rewards):
        running_return*=GAMMA
        running_return+= r
        discounted_returns.append(running_return)
    
    return list(reversed(discounted_returns))
    

if __name__ == "__main__":
    # env = gym.make("CartPole-v0")
    env_id = 'MiniGrid-Empty-5x5-v0'
    # env = gym.make(env_id)
    writer = SummaryWriter(comment="-MingiGrid Maze")

    env = gym.make(
                env_id,
                tile_size=32,
                # render_mode="human",
                agent_pov=False,
                agent_view_size=3,
                screen_size=640,
            )

    obs = env.reset()

    # print(ohc_state(env))
    # # writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PolicyGradientNet(len(ohc_state(env)), output_size=3)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)


    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_returns = [],[],[]
    episode_rewards = []

    for step_idx in range(10000):
        state = env.reset()
        state = ohc_state(env)

        total_reward = 0

        while True:

            state_v = torch.FloatTensor(state)  # convert into Float Tensor
            logits_v = net(state) # pass the state into net
            probs_v = F.softmax(logits_v, dim=0)
            action = np.random.choice(len(probs_v), p = probs_v.data.numpy()) # This causes the action to be chosen according to the probability distribution produced from the neural net.

            _, reward, done, _1, _23 = env.step(action)

            next_state = ohc_state(env)
            total_reward+=reward

            batch_states.append(state)
            batch_actions.append(action)
            episode_rewards.append(reward)


            if done:
                returns_from_episode = calculate_discounted_returns(episode_rewards)
                total_rewards.append(returns_from_episode[0])
                batch_returns.extend(returns_from_episode)
                episode_rewards.clear()
                batch_episodes+=1

                if batch_episodes>= EPISODES_TO_TRAIN:
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


            if done:
                break

            state = next_state

        # handle new rewards
        new_rewards = [total_reward]
        done_episodes += 1
        # total_rewards.append(total_reward)
        mean_rewards = float(np.mean(total_rewards[-100:]))
        print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
            step_idx, total_reward, mean_rewards, done_episodes))
        writer.add_scalar("reward", total_reward, step_idx)
        writer.add_scalar("reward_100", mean_rewards, step_idx)
        writer.add_scalar("episodes", done_episodes, step_idx)
        if mean_rewards > 195:
            print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
            break

    writer.close()



    save_model(
            net,
            'saved_model_to_show'
        )





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









