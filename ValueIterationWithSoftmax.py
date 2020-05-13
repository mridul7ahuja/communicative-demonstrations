#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import math
class ValueIteration(object):
    def __init__(self, actions, transitionPdf, rewardFunction, valueTable, convergenceTolerance, gamma, alpha, eps):
        self.actions = actions
        self.transitionPdf = transitionPdf
        self.rewardFunction  = rewardFunction
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        
    def getQvalue(self, state, action, valueTable):
        return sum((self.rewardFunction(state, action, nextState) + self.gamma*valueTable[nextState])*self.transitionPdf(state, action)[nextState]
                   for nextState in self.transitionPdf(state, action).keys())
        
    def getMaxValueAndAction(self, valueTable, state):
        maxQvalue = max(self.getQvalue(state, action, valueTable)
                               for action in self.actions)
        
        actionsProbability = {action:math.exp(self.alpha*self.getQvalue(state,action,valueTable))
                               for action in self.actions}
        
        total = sum(val for val in actionsProbability.values())
        actionsProbability = self.getPolicyDistribution(actionsProbability, lambda x: (x/total)) 
        actionsProbability = self.getPolicyDistribution(actionsProbability, lambda x: (x*(1-self.eps) + (self.eps/len(self.actions)))) 
        
        return (maxQvalue, actionsProbability) 
    
    def getPolicyDistribution(self, actionDict, func):
        return {action:func(actionDict[action]) for action in actionDict}
    
    def __call__(self):
        policyTable = {}
        stateValues = self.valueTable
        while(True):
            delta = 0
            for state,value in stateValues.items():
                temp_val = value
                if(state!=(5,2)):
                    stateValues[state], policyTable[state] = self.getMaxValueAndAction(stateValues, state)
                delta = max(delta, abs(temp_val - self.valueTable[state]))
            if(delta<=self.convergenceTolerance):
                break
        return([stateValues, policyTable])

