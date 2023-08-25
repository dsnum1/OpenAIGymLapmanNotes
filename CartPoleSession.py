import gymnasium as gym
import random
from gym.wrappers.monitoring.video_recorder import VideoRecorder
before_training = "before_training.mp4"
import time
# The following agents performs random actions


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.5):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon=epsilon
        self.random_count=0
    
    def action(self,action):
        if(random.random()<self.epsilon):
            self.random_count+=1
            print("Random: ", self.random_count)
            return self.env.action_space.sample()
        return action
    
if __name__=="__main__":
    env = RandomActionWrapper(gym.make('CartPole-v1', render_mode="rgb_array"))
    #Observation Space
    video = VideoRecorder(env, before_training)

    print(env.observation_space)      
        # Returns a box
        # low = [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
        # high= [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
        # shape=(4,)
        # dtype=float32
        # The 4 values are: x--coor, speed, angle, angular speed all wrt to centre of mass


    #Action Space
    print(env.action_space.sample())
        # Returns Discrete(2)
        # So two possible actions. Go left(0) or right(1)

    total_reward = 0
    total_steps = 0
    obs = env.reset()
    env.render()
    while True:
        video.capture_frame()
        action = env.action_space.sample()
        obs, reward, done, _ , more= env.step(action=action)
        env.render()
        total_reward+=reward
        total_steps+=1
        if(done):
            break
    print("Episode done in",total_steps,". Reward earned ",total_reward,".")



time.sleep(5)
video.close()
env.close()
