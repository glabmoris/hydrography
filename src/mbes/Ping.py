from dataclasses import dataclass

@dataclass
class Ping:
	timestamp: int  # microseconds since Unix epoch
	id: int
	quality: int
	intensity: int
	surface_sound_speed: float
	two_way_travel_time: float
	tilt_angle: float
	beam_angle: float
