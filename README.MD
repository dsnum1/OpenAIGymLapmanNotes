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


*The environment*
following members
1. action_space
2. observation_space
3. reset(): reset into initial state and return initial observation. After every episode, agenet needs to reset
4. step(): does the following
    - execute given action in the environment
    - getting new observation after this action
    - getting reward agent gained
    - getting indication of whether episode is over 
The above arguments are returned in a python tuple
(observation, reward, done, extra_info)
observation: numpy vector with observation data
reward: float value
done: boolean indicator. True or False
extra_info: could be anything. 


There are extra functions like render() also



*Creation of the environment*
There are many environments available. Named as "environment_name-vN"
Types of environments:
1. Classic Control Problems: basic problems, with low-dimension observation and action space, quick checks when implementing algorithms
2. Atari 2600
3. Algorithmic: small computation tasks, such as adding numbers
4. Board Games
5. Box2D: 2d physics walking, car control
6. Parameter unings: Optimize nn parameters
7. Toy Text: Grid world text environments
8. PyGame: Using pygame engine
9. Doom



Reward Boundary
This is the standard set to "solve" the environment. It's the average reward expected to be earned in 100 episodes to claim that your algorithm is worth it. 


Wrappers
Accumulate observations in a buffer, and provide last N observations to the agent. Pre-process an image's pixels. 
To do this, you need to redefine the methods you want to extend such as step and reset 

Subclasses of Wrappers:
1. ObservationWrapper: redefine observation
2. RewardWrapper: redefine reeward value
3. ActionWrapper: redefine action method



Monitor
Class that writes agent's performance ina  file with an option video recording of your agent in action. 