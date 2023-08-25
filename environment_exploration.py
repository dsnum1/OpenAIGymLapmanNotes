import gymnasium as gym


env    = gym.make('MiniGrid-Empty-5x5-v0')   
env.reset()

print("action_space", env.action_space)
print("observation_space", env.observation_space)
print(env.action_space.contains(-1))
print(env.action_space.sample())
print(env.observation_space['image'].low)
print(env.observation_space['image'].high)
print(env.observation_space['image'].shape)
print(env.observation_space['image'].dtype)
print(env.observation_space['image'].sample().shape)
env.render()