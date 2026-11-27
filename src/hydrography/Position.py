from dataclasses import dataclass

@dataclass
class Position:
	timestamp: int  # microseconds since Unix epoch
	latitude: float  # radians
	longitude: float  # radians
	ellipsoidal_height: float  # meters

	def __str__(self) -> str:
		return f"{self.latitude} {self.longitude} {self.ellipsoidal_height}"


