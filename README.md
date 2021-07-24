# Communicative Demonstrations
How would you change your actions if you needed to also communicate certain characteristics about them to an observer? For example, throwing your hands up in the air while cycling to demonstrate that it's the pedalling which creates most of the balance or tying your shoes slowly - with exaggerated motions - pausing at and repeating certain key actions. Demonstrative shoe-tying is very similar to ordinary shoe-tying, but also distinct. 

## Modelling using RL and CogScience
This can be rigorously modelled using techniques from Reinforcement Learning and Cognitive Science by breaking it down into three increasingly complex levels: 
1. Forming the trajectory to reach the goal state:

We use a gridworld environment with certain trap states which are sometimes safe and otherwise dangerous along with a final goal state. 
The agent has to reach the goal state with least penalties. We model the problem as a Markov Decision Process (MDP) and use value iteration to get the optimal trajectory. 

2. Predicting the environment by observing a moving agent:

When the previous agent is reaching the goal, an observer can glean insights from their movements as to which states are safe/dangerous to move on to. 
We use Probabilistic Inverse Planning and Bayesian Inference to update the belief about a certain possiblity of the environment with each step the agent takes and choose the environment with the maximum posterior probability at the goal. 

3. Reaching the goal state while taking into account observer's inverse planning:

We finally reach the communicative demonstration wherein the agent now changes his actions to maximise rewards along two goals: Reaching the end state + Communicating environment specifics to the observer. 
This is modelled as an Observer Belief MDP, and infinite continuous states are handled using discretization using k-d trees and Fitted Value Iteration using Neural Networks. 
