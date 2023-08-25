# Exploring OpenAI Gym API
### The following are notes from  from Maxim Lapan' Deep Reinforcement Learning Hands On

Objective og OpenAI Gym is to privde a rich collection of environments for RL.

`Env` --- central class called environment

All environments provide the following:
1. Set of actions allowed. This contains both discrete and continuous
2. Shape and boundaries of observations
3. Method called step to execute action. This returns observation, reward, and is_done?
4. Reset method to return the environment to its initial state and obtain the first observation. 

Action Space
1. Set of all actions for discrete action space
2. Boundaries for continuous action space


Observation Space
Pieces of information provided at every time step.

Space class
- sample(): returns a random sample from the space
- contains(x): checks if x argument belongs in the space's domain

The space class is abstract and is reimplemented in the child classes. The following are the child classes

1. Discrete(n): represents a set of items numbered from 0 to n-1. Discrete(3) = [left, right, stay]
2. Box class: n-dimensional tensor of rational numbers with intervals [low, high]. Eg. Steeering Wheel=>Discrete(low=0.0, high=1.0, shape=(1,) dtype=np.float32). Shape argument describes the shape of the tensor. Eg. In atari game, the shape=(210,160,3) describing the image input
3. Space class is a tuple class which allows several space class instances to be combined together. This is in cases where more than two actions need to be executed
4



