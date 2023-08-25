import gymnasium as gym

env = gym.make('MiniGrid-Empty-5x5-v0')
env = gym.make('ALE/Adventure-v5', render_mode="rgb_array")   
env.reset()

print("action_space", env.action_space)
print("observation_space", env.observation_space)
# print(env.action_space.contains(-1))
# print(env.action_space.sample())
# print(env.observation_space['image'].low)
# print(env.observation_space['image'].high)
# print(env.observation_space['image'].shape)
# print(env.observation_space['image'].dtype)
# print(env.observation_space['image'].sample().shape)
env.render()

# Executing an action
# print(env.step(env.action_space.sample()))
observation, reward, done, _, _1 = env.step(env.action_space.sample())

print(_1)

