import unittest
from hydrography.core import MbesReader

class TestMbesReader(unittest.TestCase):

	def test_read_kmall(self):
		reader = MbesReader()
		points = reader.read("tests/data/kmall/0006_20200917_015203_LowResPhase_subset.kmall")

		self.assertTrue( len(points) > 0 )

	def test_read_s7k(self):
		reader = MbesReader()
		points = reader.read("tests/data/s7k/20141016_150519_FJ-Saucier.s7k")
		self.assertTrue( len(points) > 0 )

	def test_read_s7k_seabat(self):
		reader = MbesReader()
		points = reader.read("tests/data/s7k/20170529_111841_Seabat.s7k")
		self.assertTrue( len(points) > 0 )

	def test_read_s7k_deneb_pds(self):
		reader = MbesReader()
		points = reader.read("tests/data/s7k/20170529_111858_VWFS Deneb_PDS.s7k")
		self.assertTrue( len(points) > 0)

	def test_read_all(self):
		reader = MbesReader()
		points = reader.read("tests/data/kongsberg_all/0008_20160909_135801_Panopee.all")
		self.assertTrue( len(points) > 0)

	def test_read_xtf_mbes(self):
		reader = MbesReader()
		points = reader.read("tests/data/xtf/0008_20160909_EM2040C_MIBAC - 0001.xtf")
		self.assertTrue( len(points) > 0)

	#'0009 - 150708_R2Testing - 0001.xtf'   Line-001-0856.sidescan.xtf   plane-sidescan2d.xtf

if __name__ == '__main__':
	unittest.main()
