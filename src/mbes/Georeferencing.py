import math
import numpy as np
from mbes.Position import Position
from mbes.Attitude import Attitude
from mbes.Ping     import Ping
from mbes.SoundVelocityProfile import SoundVelocityProfile

class Georeferencing:

	# Convert WGS84 lat/lon to ECEF
	@staticmethod
	def llh2ecef(p: Position):
		return np.array([
			(6378137.0 / (math.sqrt(1 - (0.081819190842622 ** 2) * (math.sin(p.latitude) ** 2))) + p.ellipsoidal_height) * math.cos(p.latitude) * math.cos(p.longitude),
			(6378137.0 / (math.sqrt(1 - (0.081819190842622 ** 2) * (math.sin(p.latitude) ** 2))) + p.ellipsoidal_height) * math.cos(p.latitude) * math.sin(p.longitude),
			(6378137.0 / (math.sqrt(1 - (0.081819190842622 ** 2) * (math.sin(p.latitude) ** 2))) * (1 - (0.081819190842622 **2)) + p.ellipsoidal_height) * math.sin(p.latitude)
		])

	# Computes the position of a sonar sounding in the ECEF frame
	@staticmethod
	def compute_ecef(p: Position, a: Attitude,d: Ping, svp: SoundVelocityProfile, offset: np.array, boresight: np.array):
		# Compute frame conversion matrix between local (NED) and terrestrial (ECEF) frames
		ned_2_ecef = np.array([
			[-math.cos(p.longitude)*math.sin(p.latitude),-math.sin(p.longitude),-math.cos(p.latitude)*math.cos(p.longitude)],
			[-math.sin(p.latitude)*math.sin(p.longitude),math.cos(p.longitude),-math.cos(p.latitude)*math.sin(p.longitude)],
			[math.cos(p.latitude),0,-math.sin(p.latitude)]
		])

		# Compute motion compensation matrix
		imu_2_ned  = np.array([
			[math.cos(a.heading)*math.cos(a.pitch), math.cos(a.heading)*math.sin(a.pitch)*math.sin(a.roll)-math.cos(a.roll)*math.sin(a.heading), math.cos(a.heading)*math.cos(a.roll)*math.sin(a.pitch)+math.sin(a.roll)*math.sin(a.heading)],
			[math.cos(a.pitch)*math.sin(a.heading), math.cos(a.heading)*math.cos(a.roll)+math.sin(a.pitch)*math.sin(a.roll)*math.sin(a.heading), math.sin(a.heading)*math.cos(a.roll)*math.sin(a.pitch)-math.cos(a.heading)*math.sin(a.roll)],
			[-math.sin(a.pitch),  math.cos(a.pitch) * math.sin(a.roll), math.cos(a.roll) * math.cos(a.pitch)]
		]);

		# Sum the position, offset and depth vectors in the same frame (ECEF)
		sound_speed = d.surface_sound_speed if d.surface_sound_speed > 0 else 1500.0
		distance = d.two_way_travel_time * sound_speed / 2  #TODO: raytrace with SSS + SVP

		sx = math.sin(d.tilt_angle)
		sy = math.sin(d.beam_angle)
		sz = math.sqrt(max(0.0, 1.0 - sx*sx - sy*sy))
		ping = distance * np.array([sx, sy, sz])

		#TODO: handle boresight
		return Georeferencing.llh2ecef(p) + np.matmul(np.matmul(ned_2_ecef,imu_2_ned) , (np.array([offset[0],offset[1],offset[2]]) + ping));
