import os
import sys
import pandas as pd
import numpy as np

from mbes.KmallReader import KmallReader
from mbes.S7kReader import S7kReader
from mbes.AllReader import AllReader
from mbes.XtfReader import XtfReader

from mbes.Interpolator import Interpolator, InterpolationError
from mbes.Georeferencing import Georeferencing

class MbesReader:

	def read(self,filename, offset=None, boresight=np.identity(3)):
		filename_components = os.path.splitext(filename)

		reader = False;

		if filename_components[1] =='.kmall':
			reader = KmallReader()
		elif filename_components[1] =='.s7k':
			reader = S7kReader()
		elif filename_components[1] =='.all':
			reader = AllReader()
		elif filename_components[1] =='.xtf':
			reader = XtfReader()
		else:
			raise ValueError(f"Unknown file extension {filename_components[1]}\n")

		# Read file
		reader.read(filename)

		sys.stderr.write(f"{len(reader.positions)} positions\n")
		sys.stderr.write(f"{len(reader.attitudes)} attitudes\n")
		sys.stderr.write(f"{len(reader.pings)} pings\n")

		# Interpolate time-series
		if not reader.positions or not reader.attitudes or not reader.pings:
			# Not enough data to georeference
			sys.stderr.write("Not enough data to georeference\n")
			return []

		# Sonar offset re. reference point: use the installation parameters unless given explicitly
		if offset is None:
			offset = reader.lever_arm() if hasattr(reader, "lever_arm") else np.zeros(3)
		sys.stderr.write(f"Offset (x fwd, y stbd, z down): {offset}\n")

		reader.positions.sort(key=lambda x: x.timestamp)
		reader.attitudes.sort(key=lambda x: x.timestamp)
		reader.pings.sort(key=lambda x: x.timestamp)

		# Interpolate attitudes and positions around pings
		attitude_index = 0
		position_index = 0
		n_att = len(reader.attitudes)
		n_pos = len(reader.positions)


		georeferenced_pings = []

		for ping in reader.pings:
			t = ping.timestamp

			#print(f"Ping: {ping.timestamp} Position:{reader.positions[position_index].timestamp} Attitude: {reader.attitudes[attitude_index].timestamp}")

			while attitude_index + 1 < n_att and reader.attitudes[attitude_index + 1].timestamp < t:
				attitude_index += 1

			# No more attitudes available
			if attitude_index >= n_att - 1:
				#sys.stderr.write("No more attitudes\n")
				break

			while position_index + 1 < n_pos and reader.positions[position_index + 1].timestamp < t:
				position_index += 1

			# No more positions available
			if position_index >= n_pos - 1:
				#sys.stderr.write("No more positions\n")
				break

			# No position or attitude earlier than the ping, so discard it
			if reader.positions[position_index].timestamp > t or reader.attitudes[attitude_index].timestamp > t:
				#print(f"rejecting ping {ping.id} {t} {reader.positions[position_index].timestamp} {reader.attitudes[attitude_index].timestamp}", file=sys.stderr)
				continue

			try:
				interpolated_attitude = Interpolator.interpolate_attitude(reader.attitudes[attitude_index], reader.attitudes[attitude_index + 1], t)
				interpolated_position = Interpolator.interpolate_position(reader.positions[position_index], reader.positions[position_index + 1], t)
			except InterpolationError as e:
				sys.stderr.write(f"Skipping sounding {ping.id} at {t}: {e}\n")
				continue

			#svp = self.svp_strategy.choose_svp(interpolated_position, ping)
			svp = False #TODO
			georeferenced_pings.append(Georeferencing.compute_ecef(interpolated_position, interpolated_attitude,ping, svp, offset, boresight))

		# return XYZ data
		return georeferenced_pings

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.stderr.write("Usage: MbesReader.py sonar_file\n")
		sys.exit(1)

	r = MbesReader()
	pings = r.read(sys.argv[1])

	for p in pings:
		print(f"{p[0]} {p[1]} {p[2]}")

