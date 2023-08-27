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


