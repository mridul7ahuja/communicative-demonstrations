#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math


# In[ ]:


"""
Inputs:
    action: list of possible actions
    transitionPdf: function that takes in state, action and returns a dictionary of nextState: probability
    rewardFunction: function that takes in state, action, nextState and returns a scalar reward
    valueTable: dictionary of initial values of all states
    terminalStates: list of terminalStates with their object level state as their goal state
    hyperparameters - convergenceTolerance, gamma, alpha(softmax temp.), eps(non value related randomness)
    softmax: boolean to signal softmax policy 
    
Output: 
    stateValues: dictionary of state: values
    policyTable: dictionary of state:action:probability
    
"""

class ValueIteration(object):
    def __init__(self, actions, transitionPdf, rewardFunction, valueTable, terminalStates, convergenceTolerance, gamma, alpha = 0, eps = 0, softmax = False):
        self.actions = actions
        self.transitionPdf = transitionPdf
        self.rewardFunction  = rewardFunction
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.terminalStates = terminalStates
        self.softmax = softmax
        
    def getQvalue(self, state, action, valueTable):
        return sum((self.rewardFunction(state, action, nextState) + self.gamma*valueTable[nextState])*self.transitionPdf(state, action)[nextState]
                   for nextState in self.transitionPdf(state, action).keys())
        
    def getMaxValueAndAction(self, valueTable, state):
        maxQvalue = max(self.getQvalue(state, action, valueTable) for action in self.actions)
        
        if(self.softmax):
            actionsProbability = {action:math.exp(self.alpha*self.getQvalue(state,action,valueTable))
                               for action in self.actions}
        
            total = sum(val for val in actionsProbability.values())
            actionsProbabilityNormalized = self.getPolicyDistribution(actionsProbability, lambda x: (x/total)) 
            actionsProbabilityFinal = self.getPolicyDistribution(actionsProbabilityNormalized, lambda x: (x*(1-self.eps) + (self.eps/len(self.actions))))
            
        else:
            optimalActionDict = {action:1 for action in self.actions 
                                   if(self.getQvalue(state, action, valueTable) == maxQvalue)}
            
            total = len(optimalActionDict)
            actionsProbabilityFinal = self.getPolicyDistribution(optimalActionDict, lambda x:(x/total))
        
        
        return (maxQvalue, actionsProbabilityFinal) 
    
    def getPolicyDistribution(self, actionDict, func):
        return {action:func(actionDict[action]) for action in actionDict}
    
    def __call__(self):
        policyTable = {}
        stateValues = self.valueTable
        while(True):
            delta = 0
            for state,value in stateValues.items():
                temp_val = value
                if all([state!=terminalState for terminalState in self.terminalStates]):
                    stateValues[state], policyTable[state] = self.getMaxValueAndAction(stateValues, state)
                delta = max(delta, abs(temp_val - self.valueTable[state]))
            if(delta<=self.convergenceTolerance):
                break
        return([stateValues, policyTable])

