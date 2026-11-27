from abc import ABC, abstractmethod


class BaseReader(ABC):
	def __init__(self):
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.offset    = None

	@abstractmethod
	def read(self,filename):
		pass

	def get_offset(self):
		return self.offset
