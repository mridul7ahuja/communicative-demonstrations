import sys
sys.path.append('../src/') # MODIFY TO FIT PATH

import unittest
from ddt import ddt, data, unpack

import FILENAME as targetCode # MODIFY TO FIT FILENAME OF FILE BEING TESTED


@ddt
class TestMDPConstruction(unittest.TestCase):
	def setUp(self): 
		self.dangerousColorReward = -2
		self.safeColorReward = 0
		self.goalReward = 10

		dimensions = (3,3)
		rewardScheme = {'white': self.safeColorReward , 'orange': self.dangerousColorReward, 'purple': self.dangerousColorReward, 'blue':self.safeColorReward , 'yellow':self.goalReward }
		states = {(0,0): 'white', (0,1): 'white', (0,2):'white', (1,0): 'blue', (1,1): 'purple', (1,2):'orange', (2,0): 'white', (2,1): 'yellow', (2,2):'white'}
		constructMDPFunctions = targetCode.buildMDP(dims, states, rewardScheme)

		self.getReward, self.getTransition = constructMDPFunctions()

	# BUILD MDP #######################################################################################################
	@data(ANY STATE ACTION PAIR HERE)
	@unpack
	def test_transitionFunction_checkDeterminsitic(self, state, action):
		nextStateDictionary = self.getTransition(state, action)
		numberOfNextStates = len(list(nextStateDictionary.keys()))
		self.assertEqual(numberOfNextStates, 1)

	@data(((1,1), (-1,0), (0,1)),  ((0,0), (0,1), (0,1)), ((1,1), (0,1), (1,2)))
	@unpack
	def test_transitionFunction_nonEdgeAction(self, state, action, expectedResult):
		nextStateDictionary = self.getTransition(state, action)
		deterministicNextState = list(nextStateDictionary.keys())[0]

		self.assertEqual(deterministicNextState, expectedResult)

	@data(((2,1), (1,0)),  ((0,0), (0,-1)), ((0,0), (-1,0)))
	@unpack
	def test_transitionFunction_nonEdgeAction(self, state, action):
		nextStateDictionary = self.getTransition(state, action)
		deterministicNextState = list(nextStateDictionary.keys())[0]
		expectedResult = state
		self.assertEqual(deterministicNextState, expectedResult)

	# Remaining questions - what happens when you reach terminal state? Is this something in the transition function or just in the value iteration process?
	# If it is in the value iteration only, you should put a check for it there
	# What if you put in a state not in the state set? An action?

	
	#case 1: white to purple; case 2: purple to orange; case 3: blue to purple; case 4: bounce off wall
	@data(((0,1), (1,0), (1,1)),  ((1,1), (0,1), (1,2)), ((1,0), (0,1), (1,1)), ((1,2), (0,1), (1,2)))
	@unpack
	def test_rewardFunction_dangerousColoredTile(self, state, action, nextState):
		actionTupleReward = self.getReward(state, action, nextState)
		self.assertEqual(actionTupleReward, self.dangerousColorReward)

	# Both cases to blue tile
	@data(((0,0), (1,0), (1,0)),  ((1,1), (0,-1), (1,0)))
	@unpack
	def test_rewardFunction_safeColoredTile(self, state, action, nextState):
		actionTupleReward = self.getReward(state, action, nextState)
		self.assertEqual(actionTupleReward, self.safeColorReward)

	# To white
	@data(((1,1), (-1,0), (0,1)),  ((0,0), (0,-1), (0,0)))
	@unpack
	def test_rewardFunction_whiteTile(self, state, action, nextState):
		actionTupleReward = self.getReward(state, action, nextState)
		self.assertEqual(actionTupleReward, self.safeColorReward)
	
	# To goal
	@data(((1,1), (1,0), (2,1)),  ((2,0), (0,1), (2,1)))
	@unpack
	def test_rewardFunction_goalTile(self, state, action, nextState):
		actionTupleReward = self.getReward(state, action, nextState)
		self.assertEqual(actionTupleReward, self.goalReward)


	def tearDown(self):
		pass

@ddt
class TestOBMDPConstruction(unittest.TestCase):
	def setUp(self):
		pass

	# add more testing cases here for the OBMDP when it has been written. 

	def tearDown(self):
		pass

if __name__ == '__main__':
	unittest.main(verbosity=2)