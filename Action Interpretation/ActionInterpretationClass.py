#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import functools
class actionInterpretation(object):
    def __init__(self, policiesAndMDPs, beliefPriors):
        self.policiesAndMDPs = policiesAndMDPs
        self.beliefPriors = beliefPriors
    
    def __call__(self, trajectory, world, goal):
        beliefStatePosterior = {beliefState:self.getPosterior(trajectory, beliefState) for beliefState in self.policiesAndMDPs.keys()}
        beliefVector = self.normalizePosterior(beliefStatePosterior)
        return beliefVector[(world, goal)]  
    
    def normalizePosterior(self, beliefStatePosterior):
        constant = sum(beliefStatePosterior.values())
        return {beliefState: (posterior/constant) for beliefState, posterior in beliefStatePosterior.items()}
    
    def getPosterior(self, trajectory, beliefState):
        return self.getLikelihood(trajectory, beliefState)*self.beliefPriors[beliefState]
    
    def getLikelihood(self, trajectory, beliefState):
        MDP, policy = self.policiesAndMDPs[beliefState]
        rewardFunction, transitionFunction = MDP()
        #probabilities = [policy[trajectory[i]][trajectory[i+1]]*transitionFunction[trajectory[i]][trajectory[i+1]][trajectory[i+2]] for i in range(0,len(trajectory)-1,2)]
        states = trajectory[0::2]
        actions = trajectory[1::2]
        nextStates = states[1:]
        probabilities = [policy[state][action]*transitionFunction(state,action)[nextState] for state, action, nextState in zip(states, actions, states[1:])]
        return functools.reduce(lambda a,b : a*b, probabilities)

