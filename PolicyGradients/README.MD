# Policy Gradients

An alternative way of handling MDP.
In a value based algorithm, we learn the value of every state and then act greedily in terms of value. These values are actually defining the policy. So our policy in Q Learning is a greedy policy.
`
Pi(s) = argmax*a(Q(s,a))
`

When solving RL, Policy is what we need. Obtaining value in q learning is not what is required by RL. So is it possible to skip this?

Policy is advantageous in continous action space RL problems.
Policy is advantgageous in an environment with stochasticity.


*Policy Representation*

We want our policy network to return a probability distribution of actions. 
So input state -> Neural net -> P(a= 0 ), P(a=1), P(a=2), P(a=3)... p(a=N) 

So our neural network doesn't return a single action, but rather the probability of executing each action. This allows a smooth representation. If network weights are changed even slightly, the action executed could change. 

Let's say that the output of our neural network is 40% probability for action left, and 60% probability of action right. In Q Learning or DQN, this will mean that action right is executed 100% time. But for a policy gradients agent, it means that it's 60% likely for action left to be executed while 60% probabilty of action right to be executed. 

*Policy Gradients*

How to change the network's parameters?
We will look at REINFORCE.

`PolicyGradient(PG) = -Q(s,a)log(pi(a|s))`

PG defines the direction in which we want to change our network's parameters to improve accumulated total reward. 

Scale of Gradient proportional to Q(s,a)
Gradient = log(probability of action taken).

We want to increase the probability of good actions. 
We want to decrease the probability of bad actions. 

The loss L is the same as the gradient of expected cumulative reward. 
Q(s,a) can be the same as G_t. 
log(pi(a|s)) is written as Grad(wrt. to Theta)(log(pi(a|s)))

We need to maximize the policy gradient. 


*REINFORCE method*
The above formula used to calculate policy gradients is very comminly used by most policy based methods. The way different algorithms differ is by how they define Q(s,a).

In Cross Entropy Method for example, Q(s,a) = 0 for actions from bad episodes and Q(s,a) =1 for actions from good episodes.
This is a simple execution. However, this leads to every action from bad episodes and good to be assigned similarly. There is a credit-assignment problem here. We want the actions to be rewarded according to what they deserve, their impact.

In REINFORCE, the actual Q(s,a) values are used instead of 0 and 1. This increases the probabiliy of good actions at the start.

These are steps of REINFORCE:
1. Initialize NN with random weights
2. Play N full episodes, saving their (s, a, r, s') transitions
3. For every step t of every episode k, calculate the discounted total reward for subsequent steps Q_k,t = Sum Of(Y^i*r_i)
4. Calculate the loss function for all transitions, using the equation above. Q(s,a) = Q9k,t
5. Perform the SGD update of weights minimizing loss
6. Repeat from step 2 until converged.


In the above method, exploration is performed automatically
No replay buffer is used. PG is an on-policy method.
No target network is needed. 










