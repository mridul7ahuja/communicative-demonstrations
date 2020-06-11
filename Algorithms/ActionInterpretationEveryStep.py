#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import functools


# In[ ]:


class ActionInterpretationEveryStep(object):
    def __init__(self, MDPsAndPolicies):
        self.MDPsAndPolicies = MDPsAndPolicies

    def __call__(self, trajectory, beliefPriors):
        envPosterior = {env:self.getPosteriorList(trajectory, beliefPriors, env) for env in self.MDPsAndPolicies.keys()}
        normalizedEnvPosterior = self.normalizePosterior(envPosterior)
        return(normalizedEnvPosterior)
    
    def normalizePosterior(self, envPosterior):
        envPosteriorsList = list(envPosterior.values())
        totalsList = [sum(map(lambda list: list[i], envPosteriorsList)) for i in range(len(envPosteriorsList[0]))]
        return {env: self.normalizeList(posList,totalsList) for env, posList in envPosterior.items()}
    
    def normalizeList(self, list, totalsList):
        return [y/total for y, total in zip(list, totalsList)]
    
    def getPosteriorList(self, trajectory, beliefPriors, env):
        MDP, policy = self.MDPsAndPolicies[env]
        beliefPriorsDict = beliefPriors()
        states = trajectory[0::2]
        actions = trajectory[1::2]
        nextStates = states[1:]
        probabilities = [policy[state][action]*MDP.transitionFunction(state,action)[nextState] for state, action, nextState in zip(states, actions, nextStates)]
        probAtEachStep = [functools.reduce(lambda x,y:x*y, probabilities[0:i+1])*beliefPriorsDict[env] for i in range(len(probabilities))]
        probAtEachStep.insert(0, beliefPriorsDict[env])
        return probAtEachStep
    

