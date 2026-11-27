import math

from hydrography.Position import Position
from hydrography.Attitude import Attitude

class InterpolationError(Exception):
	pass

class Interpolator:
	@staticmethod
	def interpolate_position(p1: Position, p2: Position, timestamp: int) -> Position:
		lat = Interpolator.linear_interpolation_by_time(p1.latitude, p2.latitude, timestamp, p1.timestamp, p2.timestamp)
		lon = Interpolator.linear_angle_interpolation_by_time(p1.longitude, p2.longitude, timestamp, p1.timestamp, p2.timestamp)
		alt = Interpolator.linear_interpolation_by_time(p1.ellipsoidal_height, p2.ellipsoidal_height, timestamp, p1.timestamp, p2.timestamp)

		return Position(timestamp, lat, lon, alt)

	@staticmethod
	def interpolate_attitude(a1: Attitude, a2: Attitude, timestamp: int) -> Attitude:
		roll = Interpolator.linear_angle_interpolation_by_time(a1.roll, a2.roll, timestamp, a1.timestamp, a2.timestamp)
		pitch = Interpolator.linear_angle_interpolation_by_time(a1.pitch, a2.pitch, timestamp, a1.timestamp, a2.timestamp)
		heading = Interpolator.linear_angle_interpolation_by_time(a1.heading, a2.heading, timestamp, a1.timestamp, a2.timestamp) % (2 * math.pi)

		return Attitude(timestamp, roll, pitch, heading)

	@staticmethod
	def linear_interpolation_by_time(y1: float, y2: float, x: int, x1: int, x2: int) -> float:
		"""Linear interpolation of a scalar value between (x1, y1) and (x2, y2)."""
		Interpolator.check_timestamps(x, x1, x2)
		return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

	@staticmethod
	def check_timestamps(x: int, x1: int, x2: int) -> None:
		if x1 == x2:
			raise InterpolationError("The two timestamps are the same")
		if x1 > x2:
			raise InterpolationError("The first timestamp is higher than the second timestamp")
		if not (x1 <= x <= x2):
			raise InterpolationError("The interpolation timestamp is outside the interval (extrapolation)")

	@staticmethod
	def linear_angle_interpolation_by_time(psi1: float, psi2: float, t: int, t1: int, t2: int) -> float:
		"""Linear interpolation of an angle in RADIANS along the shortest arc.
		The result is not wrapped, so a small negative roll/pitch stays a small negative number."""
		Interpolator.check_timestamps(t, t1, t2)

		delta = (t - t1) / (t2 - t1)
		dpsi = math.remainder(psi2 - psi1, 2 * math.pi)   # shortest signed difference, in [-pi, pi]

		return psi1 + dpsi * delta
