import sys
sys.path.append('../src/') # MODIFY TO FIT PATH

import unittest
from ddt import ddt, data, unpack

import FILENAME as targetCode # MODIFY TO FIT FILENAME OF FILE BEING TESTED


@ddt
class TestMDPConstruction(unittest.TestCase):
	def setUp(self): 
		pass

	## Borrow functions from the testing file I sent you - can choose a different testing example if you'd prefer (e.g. something smaller) 
	# set epsilon to 0 and beta to something large, this becomes similar to the max problem of the orifinal
	## should add a test about the behavior at the terminal site
	## Add a test for each of the parameters (temp and rationality)

	def tearDown(self):
		pass


if __name__ == '__main__':
	unittest.main(verbosity=2)