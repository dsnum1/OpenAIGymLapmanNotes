import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from boundary_env import SimpleEnv
from PolicyGradientNet import PolicyGradientNet
from save_load_model import save_model
from save_load_model import load_model


GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 3
HORIZON = 30
ENV_WIDTH=9
ENV_HEIGHT=9
ACTION_SPACE_SIZE=3
EPOCH_SIZE = 1000
N_EPOCHS= 100




class PG_trainer:
    def __init__(self):
        self.policy_net = None
        self.batch_episodes = 0


        self.episode=0
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []



        self.batch_states = []
        self.batch_actions = []
        self.batch_returns = []
        self.env = None
        self.width = ENV_WIDTH
        self.height = ENV_HEIGHT
        self.time_step_reward = 0
        self.done = False
        self.agent_x = None
        self.agent_y = None
        self.state = None
        self.optimizer = None
        self.steps = 0
        self.next_state = None
        self.action = None
        self.all_episodes_rewards = []
        self.return_at_time_step_0 = []
        self.successful_episodes  = 0


    def set_env(self):
        self.env = SimpleEnv(   # initialize the environment
            width=ENV_WIDTH,
            height=ENV_HEIGHT
        )

        self.env.reset()   # initialize the environment
    
    def env_step(self, action):
        self.action = action
        _, self.time_step_reward, self.done, _1, _23 = self.env.step(action)
        
    
    def get_one_hot_encoded_state(self):
        self.agent_x = self.env.agent_pos[0]
        self.agent_y = self.env.agent_pos[1]
        return self.env.get_state(self.agent_x, self.agent_y)

    def calculate_discounted_returns(self):
        discounted_returns = []
        running_return = 0.0

        for r in reversed(self.episode_rewards):
            running_return*=GAMMA
            running_return+= r
            discounted_returns.append(running_return)
        self.episode_discounted_returns = list(reversed(discounted_returns))
        return list(reversed(discounted_returns))
    

    def setup_new_policy_net(self):
        self.state = self.get_one_hot_encoded_state()

        self.policy_net = PolicyGradientNet(
            input_size=len(self.state),
            output_size=ACTION_SPACE_SIZE
        )   
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)


    def choose_action(self, probs_v):
        action = np.random.choice(len(probs_v), p = probs_v.data.numpy())
        return action

    
    def append_to_episode_history(self):
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.time_step_reward)

    def clear_episode_history(self):
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_discounted_returns.clear()

    def extend_batch(self):
        self.batch_states.extend(self.episode_states)
        self.batch_actions.extend(self.episode_actions)
        self.batch_returns.extend(self.calculate_discounted_returns())

    def clear_batches(self):
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_returns.clear()


    def play_1_episode(self):
        self.env.reset()
        self.steps = 0
        self.episode_rewards = []
        self.state = self.get_one_hot_encoded_state()
        
        while self.steps < HORIZON:
            self.steps+=1
            state_tensor = torch.FloatTensor(self.state)
            logits_v = self.policy_net(state_tensor)
            probs_v = F.softmax(logits_v, dim=0)
            action = self.choose_action(probs_v)
            self.env_step(action)
            self.next_state = self.get_one_hot_encoded_state()
            
            self.append_to_episode_history()

            if self.done == True:
                self.successful_episodes+=1
                break


        self.episode_end_sequence()

    

        return "Episode Ended"
    

    def episode_end_sequence(self):
        self.episode+=1
        self.batch_episodes+=1 


        self.calculate_discounted_returns() # Calculate discounted rewards
        self.extend_batch()
        # self.return_at_time_step_0.append(self.episode_discounted_returns[0]) # what's the return at time step 1
        # print('Episode number: ', self.episode, ' return at time step 0', self.episode_discounted_returns[0])
        
        self.clear_episode_history()
        self.handle_batch_learning()

        pass
    
    def progress_report(self):
        # plot frequency plot of successful episodes verses number of total episodes
        # for all successful episodes plot the number of steps required to reach the map
        # for all successful episodes plot the return at time step 0

        # heatmap of all the states covered 
        pass


    def handle_batch_learning(self):
        if(self.batch_episodes<4):
            return 
        states_v = torch.FloatTensor(self.batch_states)
        batch_actions_t = torch.LongTensor(self.batch_actions)
        batch_returns_v = torch.FloatTensor(self.batch_returns)

        self.optimizer.zero_grad()
        logits_v = self.policy_net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim = 1)
        log_prob_actions_v = batch_returns_v * log_prob_v[range(len(self.batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()
        loss_v.backward()
        self.optimizer.step()


        self.batch_episodes = 0
        self.clear_batches()        

        pass

    def train(self):
        for i in range(2000000):
            self.play_1_episode()
            if (i%100 == 0):
                torch.save(self.policy_net.state_dict, 'classed_PG.pth')




a = PG_trainer()
a.set_env()
a.setup_new_policy_net()
a.train()






