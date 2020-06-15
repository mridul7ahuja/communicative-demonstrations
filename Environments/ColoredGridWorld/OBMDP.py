#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import spatial
import sys
sys.path.append('../Environments/')
from ColoredGridWorld import SetUpInferenceSpace as setUp


# In[ ]:


class OBMDP(object):
    def __init__(self, trueEnv, isInformativeBeliefGoal, beta, objectTransitionFn, getNextBelief):
        self.trueEnv = trueEnv
        self.isInformativeBeliefGoal = isInformativeBeliefGoal
        self.beta = beta
        self.objectTransitionFn = objectTransitionFn
        self.getNextBelief = getNextBelief
        
    def __call__(self, objectRewardFn, beliefUtilityFn):
        return self.getRewardFunction(objectRewardFn, beliefUtilityFn)
    
    def getRewardFunction(self, objectRewardFn, beliefUtilityFn):
        return lambda jointState, action, nextJointState: objectRewardFn(jointState[0], action , nextJointState[0]) + self.beta*beliefUtilityFn(jointState[1](), nextJointState[1](), self.trueEnv, self.isInformativeBeliefGoal)
        
    def transitionFunction(self, jointState, action):
        return {(nextObjectState, self.getNextBelief(jointState, action, nextObjectState)): self.objectTransitionFn(jointState[0], action)[nextObjectState] for nextObjectState in self.objectTransitionFn(jointState[0], action).keys()}

    """
    
        Class to get the literal observer function depending on the dicretization of the belief space
        Input: 
            __init__:
            actionInterpretation: object of class ActionInterpretation 
            __call__
            beliefSpace: list of dictionaries of this type - {(env):probability}
            isDiscretized: boolean to indicate whether this should assume the belief space is discretized
        Output: 
            an indicator function that takes as input jointState, action, nextObjectState and returns the next observer belief state 
    """
    
class LiteralObserver(object):
    def __init__(self, actionInterpretation, discreteBeliefSpace):
        self.actionInterpretation = actionInterpretation
        self.discreteBeliefSpace = discreteBeliefSpace
    
    def getNextDiscreteBelief(self, jointState, action, nextObjectState):
        return (self.getNearestNeighbour(self.actionInterpretation( [jointState[0], action, nextObjectState] , jointState[1])))
        
    def getNextBelief(self, jointState, action, nextObjectState):
        return (self.actionInterpretation( [jointState[0], action, nextObjectState], jointState[1]))
    
    #finds the nearest neighbour of the observer's new belief in the belief space
    def getNearestNeighbour(self, observerNewBelief):
        observerNewBeliefDict = observerNewBelief()
        envOrder = tuple(env for env in observerNewBeliefDict.keys())
        observerNewBeliefVector = [observerNewBeliefDict[env] for env in envOrder]
        discreteBeliefSpaceVectors = [[belief[env] for env in envOrder] for belief in self.discreteBeliefSpace] 
        tree = spatial.cKDTree(discreteBeliefSpaceVectors)
        nearestNeighbourVector = discreteBeliefSpaceVectors[ tree.query(observerNewBeliefVector)[1] ]
        nearestNeighbour = {env:probability for env, probability in zip(envOrder, nearestNeighbourVector)}
        hashableNearestNeighbour = setUp.HashableBelief(nearestNeighbour)
        return hashableNearestNeighbour 

#Equation 10 in Ho et al. 
def getBeliefUtility():
    return lambda belief, nextBelief, trueEnv, isInformativeBeliefGoal: nextBelief[trueEnv] - belief[trueEnv] if (isInformativeBeliefGoal) else 0

