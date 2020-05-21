#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class MDP(object):
    def __init__(self, dimensions, stateSpace, colourReward):
        self.dimensions = dimensions
        self.stateSpace = stateSpace
        self.colourReward = colourReward
        
    def buildRewardFunction(self):
        transitionPdf = self.buildTransitionFunction()
        return lambda s, a, nextS: transitionPdf(s,a)[nextS]*self.colourReward[self.stateSpace[nextS]] 
        
    def buildTransitionFunction(self): 
        n,m = self.dimensions
        return lambda s, a: {((s[0] + a[0]), (s[1] + a[1])) : 1.0} if( ((s[0] + a[0])<m and (s[1] + a[1])<n) and ((s[0] + a[0])>=0 and (s[1] + a[1])>=0) ) else {s:1.0}
    
    
    def __call__(self):
        return ([self.buildRewardFunction(), self.buildTransitionFunction()])

