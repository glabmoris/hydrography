from abc import ABC, abstractmethod

class BaseReader(ABC):
	def __init__(self):
		self.positions = []
		self.attitudes = []
		self.pings     = []

	@abstractmethod
	def read(self,filename):
		pass
