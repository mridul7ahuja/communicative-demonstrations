#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class ValueIteration(object):
    def __init__(self, actions, transitionPdf, rewardFunction, valueTable, convergenceTolerance, gamma):
        self.actions = actions
        self.transitionPdf = transitionPdf
        self.rewardFunction  = rewardFunction
        self.valueTable = valueTable
        self.convergenceTolerance = convergenceTolerance
        self.gamma = gamma
        
    def getQvalue(self, state, action, valueTable):
        return sum((self.rewardFunction(state, action, nextState) + self.gamma*valueTable[nextState])*self.transitionPdf(state, action)[nextState]
                   for nextState in self.transitionPdf(state, action).keys())
        
    def getMaxValueAndAction(self, valueTable, state):
        maxQvalue = max(self.getQvalue(state, action, valueTable)
                               for action in self.actions)
        
        optimalActionDict = {action:1 for action in self.actions 
                               if(self.getQvalue(state, action, valueTable) == maxQvalue) }
        
        self.getPolicyDistribution(optimalActionDict, lambda x:1/(len(x)))
        return (maxQvalue, optimalActionDict) 
    
    def getPolicyDistribution(self, actionDict, func):
        for action in actionDict:
            actionDict[action] = func(actionDict)
    
    def __call__(self):
        policyTable = {}
        stateValues = self.valueTable
        while(True):
            delta = 0
            for state,value in stateValues.items():
                temp_val = value
                stateValues[state], policyTable[state] = self.getMaxValueAndAction(stateValues, state)
                delta = max(delta, abs(temp_val - self.valueTable[state]))
            if(delta<=self.convergenceTolerance):
                break
        #print("mera khudka valueiteration")
        return([stateValues, policyTable])

