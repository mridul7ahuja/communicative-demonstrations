#!/usr/bin/env python
# coding: utf-8

# In[4]:


from itertools import product
import sys
sys.path.append('../Environments/')
from ColoredGridWorld.MDP import MDP
from ColoredGridWorld import OBMDP
sys.path.append('../Algorithms/')
from ValueIteration import ValueIteration
from DictValueIteration import ValueIteration as DictValueIteration
import math 
import copy


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
    rewardFunctions = [MDP() for MDP in envMDPs]
    transitionFunctions = [MDP.transitionFunction for MDP in envMDPs]
    valueTable = {key: 0 for key in stateSpace.keys()}
    ValueIterations = [ValueIteration(actions, transitionFunction, rewardFunction, valueTable, [env[1]], convergenceTolerance, gamma, alpha, eps, True) for rewardFunction, transitionFunction, env in zip(rewardFunctions, transitionFunctions,envSpace)]
    ValueAndPolicyTables = [ValueIteration() for ValueIteration in ValueIterations]
    envPolicies = [ValueAndPolicyTable[1] for ValueAndPolicyTable in ValueAndPolicyTables]
    return {(env):(MDP,policy) for env, MDP, policy in zip(envSpace, envMDPs, envPolicies)}

"""
    Constructs mapping of envs to their OBMDPs and Policies  
    Input:
        jointStateSpace: list of all possible (objectState, HashableBelief) tuples
        actions: list of actions
        envMDPsAndPolicies: dictionary of all env:(MDP, Policy)
        getNextBelief: function that inputs (jointState, action, nextObjectState) and returns the next belief state
        beliefHyperparameters: list of convergenceTolerance, beta, gamma(discount factor), alpha(softmax temp), eps(non-value randomness)
        
    Output: 
        dictionary of all env:(OBMDP, Policy)

"""

def buildPragmaticEnvPolicySpace(jointStateSpace, actions, envMDPsAndPolicies, getNextBelief, beliefHyperparameters):
    convergenceTolerance, beta, beliefGamma, beliefAlpha, beliefEps = beliefHyperparameters
    beliefUtilityFn = OBMDP.getBeliefUtility()
    envOrder = tuple(env for env in envMDPsAndPolicies.keys())
    envOBMDPs = [OBMDP.OBMDP(env, True, beta, envMDPsAndPolicies[env][0].transitionFunction, getNextBelief) for env in envOrder]
    jointRewardFunctions = [OBMDP(envMDPsAndPolicies[env][0](), beliefUtilityFn) for env, OBMDP in zip(envOrder, envOBMDPs)]
    jointTransitionDicts = [{jointState:{action:OBMDP.transitionFunction(jointState, action) for action in actions} for jointState in jointStateSpace if jointState[0]!=env[1]} for OBMDP, env in zip(envOBMDPs, envOrder)]
    jointRewardDicts = []
    for jointTransitionDict, jointRewardFn in zip(jointTransitionDicts, jointRewardFunctions):
        jointRewardDict = copy.deepcopy(jointTransitionDict)
        for jointState, actionAndNextStateDict in jointRewardDict.items():
            for action, nextJointStateAndProb in actionAndNextStateDict.items():
                for nextJointState in nextJointStateAndProb.keys():
                    jointRewardDict[jointState][action][nextJointState] = jointRewardFn(jointState, action, nextJointState)
        jointRewardDicts.append(jointRewardDict)
    valueTable = {key: 0 for key in jointStateSpace}
    jointGoalStatesList = [[jointState for jointState in jointStateSpace if jointState[0] == env[1]] for env in envOrder]
    ValueIterations = [DictValueIteration(jointTransitionDict, jointRewardDict, valueTable, jointGoalStates, convergenceTolerance, beliefGamma, beliefAlpha, beliefEps) for jointTransitionDict, jointRewardDict, jointGoalStates in zip(jointTransitionDicts, jointRewardDicts, jointGoalStatesList)]
    ValueAndPolicyTables = [ValueIteration() for ValueIteration in ValueIterations]
    envPolicies = [ValueAndPolicyTable[1] for ValueAndPolicyTable in ValueAndPolicyTables]
    return {env:(OBMDP, policy) for env, OBMDP, policy in zip(envOrder, envOBMDPs, envPolicies)}
    
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

"""
    Given a dictionary of env (i.e. (world, goal) tuple):_ returns a dictionary with envs replaced with their x (trap) o (safe) labels
    Input:
        envDict: dict of the form env:_
        orderOfTrapColours: list/tuple of the order of potential trap states for the label ordering 
                            for eg. when orange and purple are traps with order being (orange, purple, blue), label will be xxo
    Output:
        dictionary with envs replaced with desired labels
"""

def mapEnvToLabel(envDict, orderOfTrapColours):
    envLabels = {}
    for env in envDict.keys():
        colourReward = env[0]()
        label = ""
        for colour in orderOfTrapColours:
            if(colourReward[colour]<0):
                label = label + "x"
            else: 
                label = label + "o"
        envLabels[env] = label
    return {envLabels[env]:prob for env, prob in envDict.items()} 

