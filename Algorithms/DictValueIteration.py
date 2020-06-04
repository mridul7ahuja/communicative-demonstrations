#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import pprint


# In[ ]:


class ValueIteration(object):
    def __init__(self, transitionTable, rewardTable, valueTable, terminalStates, convergenceTolerance, gamma, alpha = 0, eps = 0):
        self.transitionTable = transitionTable
        self.rewardTable  = rewardTable
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.terminalStates = terminalStates
        
    def getQvalue(self, action, nextStatesAndRewards, valueTable, nextStatesAndProbabilities):
        return sum((reward + self.gamma*valueTable[nextState])*nextStatesAndProbabilities[action][nextState]
                   for nextState, reward in nextStatesAndRewards.items())
        
    def getMaxValueAndAction(self, valueTable, state):
        nextStatesAndProbabilities = self.transitionTable[state]
        actionsAndRewards = self.rewardTable[state]
        maxQvalue = max(self.getQvalue(action,nextStatesAndRewards,valueTable,nextStatesAndProbabilities)
                               for action, nextStatesAndRewards in actionsAndRewards.items())
        
        actionsProbability = {action:math.exp(self.alpha*self.getQvalue(action,nextStatesAndRewards,valueTable,nextStatesAndProbabilities)) 
                               for action, nextStatesAndRewards in actionsAndRewards.items()}
        
        total = sum(val for val in actionsProbability.values())
        actionsProbabilityNormalized = self.getPolicyDistribution(actionsProbability, lambda x: (x/total)) 
        actionsProbabilityFinal = self.getPolicyDistribution(actionsProbabilityNormalized, lambda x: (x*(1-self.eps) + (self.eps/len(actionsProbability.keys()))))
        
        return (maxQvalue, actionsProbabilityFinal) 
    
    def getPolicyDistribution(self, actionDict, func):
        return {action:func(actionDict[action]) for action in actionDict}
    
    def __call__(self):
        #######################################
        ########## YOUR CODE HERE #############
        #######################################
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

