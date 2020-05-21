#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import functools


# In[ ]:


"""
Input: 
    __init__:
        MDPsAndPolicies: dictionary of env:(MDP,Policy)
        
    __call__: 
        trajectory: list of state, action, nextState
        beliefPriors: dictionary of env:prior probabilty
        world: world object
        goal: object level goal state
        vector: boolean value to return the whole belief vector or else a particular environment's probability
        
Output:
    beliefVector: dictionary of env:probability
    OR
    Probability of particular environment
"""


class ActionInterpretation(object):
    def __init__(self, MDPsAndPolicies):
        self.MDPsAndPolicies = MDPsAndPolicies
    
    def __call__(self, trajectory, beliefPriors, world = [], goal = [], vector = True):
        beliefStatePosterior = {env:self.getPosterior(trajectory, beliefPriors, env) for env in self.MDPsAndPolicies.keys()}
        beliefVector = self.normalizePosterior(beliefStatePosterior)
        if(vector):
            return beliefVector
        else: 
            return beliefVector[(world, goal)]  
    
    def normalizePosterior(self, beliefStatePosterior):
        constant = sum(beliefStatePosterior.values())
        return {env: (posterior/constant) for env, posterior in beliefStatePosterior.items()}
    
    def getPosterior(self, trajectory, beliefPriors, env):
        return self.getLikelihood(trajectory, env)*beliefPriors[env]
    
    def getLikelihood(self, trajectory, env):
        MDP, policy = self.MDPsAndPolicies[env]
        rewardFunction, transitionFunction = MDP()
        states = trajectory[0::2]
        actions = trajectory[1::2]
        nextStates = states[1:]
        probabilities = [policy[state][action]*transitionFunction(state,action)[nextState] for state, action, nextState in zip(states, actions, states[1:])]
        return functools.reduce(lambda a,b : a*b, probabilities)

