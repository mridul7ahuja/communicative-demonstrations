#!/usr/bin/env python
# coding: utf-8

# In[4]:


from itertools import product
import sys
sys.path.append('../Environments/')
from ColoredGridWorld.MDP import MDP
sys.path.append('../Algorithms/')
from ValueIteration import ValueIteration
import math 


# In[ ]:


class HashableBelief(object):
    def __init__(self, beliefDict):
        self.beliefDict = beliefDict
    
    def __call__(self):
        return self.beliefDict
    
    def __eq__(self, other): 
        if not isinstance(other, HashableBelief):
            return NotImplemented

        return self.beliefDict == other.beliefDict 
    
    def __hash__(self):
        return hash((frozenset(self.beliefDict.items())))


class HashableWorld(object):
    def __init__(self, colourReward, isDeterministic):
        self.colourReward = colourReward
        self.isDeterministic = isDeterministic
        
    def __call__(self):
        return (self.colourReward)
    
    def __eq__(self, other): 
        if not isinstance(other, HashableWorld):
            return NotImplemented

        return self.colourReward == other.colourReward and self.isDeterministic == other.isDeterministic
    
    def __hash__(self):
        return hash((frozenset(self.colourReward.items()), self.isDeterministic))
    
"""
    Constructs all possible colour reward mappings 
    Input: 
        variableColours: list of colours with variable rewards
        variableReward: list of possible scalar rewards for the variable colours
        constantRewardDict: dict of colour: reward for colours whose rewards are fixed
    Output:
        utilitySpace: list of dictionaries corresponding to all possible colours reward mappings

"""
    
def buildUtilitySpace(variableColours, variableReward, constantRewardDict):
    utilitySpace = [{key:value for key, value in zip(variableColours, permutations)} for permutations in product(variableReward, repeat = len(variableColours))]
    for i in range(len(utilitySpace)):
        utilitySpace[i].update(constantRewardDict)
    return utilitySpace

"""
    Constructs mapping of envs to their MDPs and Policies  
    Input:
        dimensions: tuple of mXn
        stateSpace: dict of state: colours
        actions: list of actions
        envSpace: list of (world,goal) tuples
        hyperparameters: list of convergenceTolerance, gamma(discount factor), alpha(softmax temp), eps(non-value randomness)
        
    Output: 
        dictionary of all env:(MDP, Policy)
"""

def buildEnvPolicySpace(dimensions, stateSpace, actions, envSpace, hyperparameters):
    convergenceTolerance, gamma, alpha, eps = hyperparameters 
    envMDPs = [MDP(dimensions, stateSpace, world()) for world,goal in envSpace] 
    rewardAndTransitionFunctions = [MDP() for MDP in envMDPs]
    rewardFunctions = [rewardAndTransitionFunction[0] for rewardAndTransitionFunction in rewardAndTransitionFunctions]
    transitionFunctions = [rewardAndTransitionFunction[1] for rewardAndTransitionFunction in rewardAndTransitionFunctions]
    valueTable = {key: 0 for key in stateSpace.keys()}
    ValueIterations = [ValueIteration(actions, transitionFunction, rewardFunction, valueTable, [env[1]], convergenceTolerance, gamma, alpha, eps, True) for rewardFunction, transitionFunction, env in zip(rewardFunctions, transitionFunctions,envSpace)]
    ValueAndPolicyTables = [ValueIteration() for ValueIteration in ValueIterations]
    envPolicies = [ValueAndPolicyTable[1] for ValueAndPolicyTable in ValueAndPolicyTables]
    return {(env):(MDP,policy) for env, MDP, policy in zip(envSpace, envMDPs, envPolicies)}

"""
    Returns list of all possible worlds
    Input: 
        utilitySpace: list of possible colour:reward dictionaries
        transitionSpace: list of boolean values (True for deterministic, False for not)
    Output: 
        list of World objects
"""

def buildWorldSpace(utilitySpace, transitionSpace):
    return [HashableWorld(colourReward, isDeterministic) for colourReward, isDeterministic in product(utilitySpace, transitionSpace)]

