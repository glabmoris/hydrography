import math
import pandas as pd
import re
from datetime import datetime

class SoundVelocityProfileSection:
	def __init__(self,timestamp,longitude,latitude):
		self.longitude = longitude
		self.latitude  = latitude
		self.timestamp = timestamp
		self.values = pd.DataFrame(columns=["depth","sound_speed"])

	def get_values(self):
		return self.values

	def get_position(self):
		return (self.longitude,self.latitude)

	def get_timestamp(self):
		return self.timestamp

	def add(self,depth,sound_speed):
		self.values.loc[len(self.values)] = [depth,sound_speed]

class SoundVelocityProfile:
	def __init__(self):
		self.sections = []
		self.name   = ""

	def dms2dd(self,dms_string):
		degrees,minutes,seconds = re.split("[:]",dms_string)
		dd = abs(float(degrees)) + (float(minutes)/60.0) + (float(seconds)/(60.0*60.0))
		return dd if float(degrees) > 0 else -1 * dd

	def read(self,filename):
		with open(filename) as f:
			line_count = 0

			for line in f:
				if line_count == 0:
					if not line.startswith("[SVP_VERSION_2]"):
						raise Exception("Unsupported SVP file version: only version 2 is supported")
				elif line_count == 1:
					name=line
				else:
					tokens = line.split()

					if tokens[0]=="Section":
						timestamp = datetime.strptime(tokens[1] + " " + tokens[2],"%Y-%j %H:%M:%S")
						try:
							latitude = self.dms2dd(tokens[3])
						except Exception as e:
							print(f"{e}")
							latitude = math.nan
						try:
							longitude = self.dms2dd(tokens[4])
						except Exception as e:
							longitude = math.nan

						#new section
						new_section = SoundVelocityProfileSection(timestamp,longitude,latitude)
						self.sections.append(new_section)
					else:
						depth = float(tokens[0])
						sound_speed = float(tokens[1])

						self.sections[len(self.sections)-1].add(depth,sound_speed)

				line_count = line_count + 1
	def get_sections(self):
		return self.sections

