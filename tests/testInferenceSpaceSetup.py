import sys
sys.path.append('../src/') # MODIFY TO FIT PATH

import unittest
from ddt import ddt, data, unpack
import pandas as pd

import FILENAME as targetCode # MODIFY TO FIT FILENAME OF FILE BEING TESTED


@ddt
class TestInferenceSpaceSetup(unittest.TestCase):
	def setUp(self): 
		pass

	# buildWorld #######################################################################################################
	@data()
	@unpack
	def test_buildWorld_checkHashing(self,):
		# make sure the hashing functions works the way you mean it to work
		# consider renaming buildWorld to mention that it can be called as a key in a dictionary like buildHashableWorld or transformEnvironmentParametersToKey
		pass

	# getUtilitySpace #######################################################################################################
	@data()
	@unpack
	def test_utilitySpace_checkSizeOfUtilitySpace(self,):
		# make sure the the size of space is correct
		pass

	@data()
	@unpack
	def test_utilitySpace_itemInUtilitySpace(self,):
		# make sure a particular item is in there (and that you can call it properly)
		pass

	# getWorldSpace #######################################################################################################
	# similar to utility space

	# getEnvironmentPolicySpace #######################################################################################################
	# similar to utility space



	def tearDown(self):
		pass

if __name__ == '__main__':
	unittest.main(verbosity=2)