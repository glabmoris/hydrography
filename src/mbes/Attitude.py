from dataclasses import dataclass

@dataclass
class Attitude:
	timestamp: int  # microseconds since Unix epoch
	roll: float  # radians
	pitch: float  # radians
	heading: float  # radians

