import sys
sys.path.append('../src/') # MODIFY TO FIT PATH

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import OTHERFILES #anything else you need to build the test cases
import FILENAME as targetCode # MODIFY TO FIT FILENAME OF FILE BEING TESTED


@ddt
class TestActionInterpretation(unittest.TestCase):
	def setUp(self): 
		# SEE the email attachment for world/trajectory/tests
		# build the inputs and everything needed to create the inputs
		# beliefPoliciesAndMDPs = buildPoliciesAndMDPs(dimensions, stateSpace, utilitySpace, beliefSpace, actions, valueTable, hyperparameters)
		# uniformPrior = {(belief): (1/len(beliefSpace)) for belief in beliefSpace}
		# nonUniformPrior = 
		# self.beliefVector = actionInterpretation(beliefPoliciesAndMDPs, priors) 

	# Action Interpretion #######################################################################################################
	# example test
	@data()
	@unpack
	def test_actionInterpretationPosterior_equivalentEnvironmentBeliefs(self, belief1, belief2):
		#set up the correct combination of worlds trajectories and environments
		self.assertAlmostEqual(posteriorProbabilityOfBelief2, posteriorProbabilityOfBelief2)




	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest.main(verbosity=2)