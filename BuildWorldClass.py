#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class buildWorld(object):
    def __init__(self, colourReward, isDeterministic):
        self.colourReward = colourReward
        self.isDeterministic = isDeterministic
        
    def __call__(self):
        return (self.colourReward)
    
    def __eq__(self, other): 
        if not isinstance(other, buildWorld):
            return NotImplemented

        return self.colourReward == other.colourReward and self.isDeterministic == other.isDeterministic
    
    def __hash__(self):
        return hash((frozenset(self.colourReward.items()), self.isDeterministic))

