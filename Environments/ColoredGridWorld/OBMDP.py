#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import spatial
import sys
sys.path.append('../Environments/')
from ColoredGridWorld import SetUpInferenceSpace as setUp


# In[ ]:


class OBMDP(object):
    def __init__(self, jointStateSpace, trueEnv, isInformativeBeliefGoal, beta):
        self.jointStateSpace = jointStateSpace
        self.trueEnv = trueEnv
        self.isInformativeBeliefGoal = isInformativeBeliefGoal
        self.beta = beta
        
    def __call__(self,objectTransitionFn, objectRewardFn, getNextBelief, beliefUtilityFn):
        return [self.getRewardFunction(objectRewardFn, beliefUtilityFn), self.getTransitionFunction(objectTransitionFn, getNextBelief)]
    
    def getRewardFunction(self, objectRewardFn, beliefUtilityFn):
        return lambda jointState, action, nextJointState: objectRewardFn(jointState[0], action , nextJointState[0]) + self.beta*beliefUtilityFn(jointState[1](), nextJointState[1](), self.trueEnv, self.isInformativeBeliefGoal)
        
    def getTransitionFunction(self, objectTransitionFn, getNextBelief):
        return lambda jointState, action: {(nextObjectState, getNextBelief(jointState, action, nextObjectState)): objectTransitionFn(jointState[0], action)[nextObjectState] for nextObjectState in objectTransitionFn(jointState[0], action).keys()}
    
    """
        Class to get the literal observer function depending on the dicretization of the belief space
        Input: 
            __init__:
            actionInterpretation: object of class ActionInterpretation 
            __call__
            beliefSpace: list of dictionaries of this type - {(env):probability}
            isDiscretized: boolean to indicate whether this should assume the belief space is discretized
        Output: 
            an indicator function that takes as input jointState, action, nextObjectState and returns the required belief state
    """
    
class LiteralObserver(object):
    def __init__(self, actionInterpretation):
        self.actionInterpretation = actionInterpretation
    
    def __call__(self, discreteBeliefSpace = [], isDiscretized = False):  
        if(isDiscretized):
            return lambda jointState, action, nextObjectState : (self.getNearestNeighbour(self.actionInterpretation( [jointState[0], action, nextObjectState] , jointState[1]), discreteBeliefSpace))
        else:
            return lambda jointState, action, nextObjectState : (self.actionInterpretation( [jointState[0], action, nextObjectState], jointState[1]))
    
    #finds the nearest neighbour of the observer's new belief in the belief space
    def getNearestNeighbour(self, observerNewBelief, discreteBeliefSpace):
        observerNewBeliefDict = observerNewBelief()
        envOrder = tuple(env for env in observerNewBeliefDict.keys())
        observerNewBeliefVector = [observerNewBeliefDict[env] for env in envOrder]
        discreteBeliefSpaceVectors = [[belief[env] for env in envOrder] for belief in discreteBeliefSpace] 
        tree = spatial.cKDTree(discreteBeliefSpaceVectors)
        nearestNeighbourVector = discreteBeliefSpaceVectors[ tree.query(observerNewBeliefVector)[1] ]
        nearestNeighbour = {env:probability for env, probability in zip(envOrder, nearestNeighbourVector)}
        hashableNearestNeighbour = setUp.HashableBelief(nearestNeighbour)
        return hashableNearestNeighbour 

#Equation 10 in Ho et al. 
def getBeliefUtility():
    return lambda belief, nextBelief, trueEnv, isInformativeBeliefGoal: nextBelief[trueEnv] - belief[trueEnv] if (isInformativeBeliefGoal) else 0

