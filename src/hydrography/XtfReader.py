import struct
import mmap
import sys
import math
import datetime
import statistics
import numpy as np

from hydrography.BaseReader import BaseReader
from hydrography.Ping import Ping
from hydrography.Attitude import Attitude
from hydrography.Position import Position

# References are to the Triton Imaging "eXtended Triton Format (XTF) Rev. 41" (September 2016). XTF values are little
# endian (2.5) with 1 byte packing (2.2.1). Reson 7125 bathymetry is stored as a raw 7027 record (appendix 1), whose
# layout is from the Teledyne RESON "7k Data Format, Volume I" version 2.41, which appendix 1 refers to.

# Two misprints in the XTF document: Table E gives HeaderType at offset 1 and SubChannelNumber at 2, and Table H gives
# ReservedSpace2 at 245. Every packet starts with the same 14 bytes (magic 0, type 2, subchannel 3, size 10), and the
# stated 64 and 256 byte totals only work out with ReservedSpace2 at 250, so those are used here.

FILE_FORMAT       = 123			# Table C: FileFormat
MAGIC_NUMBER      = 0xFACE
MAGIC_BYTES       = struct.pack("<H", MAGIC_NUMBER)
FILE_HEADER_BLOCK = 1024		# the file header grows in 1024 byte increments for more than six channels (2.4.1)
NAV_UNITS_LATLONG = 3			# Table C: NavUnits, 0 = meters (projected), 3 = lat/long
TYPE_BATHYMETRY   = 3			# Table D: TypeOfChannel

FILE_HEADER        = struct.Struct("<BB8s8s16sH64s64sHHHBBHBBHf12s10si12f")	# Table C up to the CHANINFO array, 256 bytes
CHANINFO           = struct.Struct("<BBHHHI16s11fHB53s")	# Table D, 128 bytes
PACKET_HEADER      = struct.Struct("<HBBHHHI")			# common start of all packets: magic, type, subchannel, channels, reserved, size
PING_HEADER        = struct.Struct("<HBBHHHIHBBBBBBHIIffI21fddHHBBBBffddHHHH10fIfIBBBBhhBIB6s")	# Table H, 256 bytes
ATTITUDE           = struct.Struct("<HBBHHHIIIIIffffIfHBBBBBHB")	# Table E, 64 bytes
NAVIGATION         = struct.Struct("<HB7sIHBBBBBIIIdddB6s")		# 3.1.18, 64 bytes
POS_RAW_NAVIGATION = struct.Struct("<HBBHHHIHBBBBBHdddffffB")	# Table P, 64 bytes
Q_MULTIBEAM_ENTRY  = struct.Struct("<ididdddffff")				# Table S (XTFQPSMBEENTRY), 64 bytes

# Reson 7k structures (7k DFD tables 5, 66 and 67)
DRF_7K             = struct.Struct("<HHIIIIHHfBBHIIHHIHHIII")	# Data Record Frame, 64 bytes
SYNC_PATTERN_7K    = 0x0000FFFF
RAW_DETECTION_7027 = struct.Struct("<QIHIIBIff16I")
DETECTION_7027     = struct.Struct("<HffIIff")		# may be shorter (older) or longer (newer): step by the RTH data field size
DETECTION_7027_MIN = struct.calcsize("<Hff")			# beam descriptor, detection point and rx angle are required
RECORD_7006        = 7006
RECORD_7027        = 7027
UNCERTAINTY_METHOD_MASK  = 0x0F	# 7027 RTH flags bits 0-3, 0 = uncertainty not calculated
DETECTION_PRIORITY_SHIFT = 9		# 7027 detection flags bits 9-12, multi-detect priority, 0 = highest
DETECTION_PRIORITY_MASK  = 0x0F

# Header types (Table H and sections 3.1.13-3.1.19)
HEADER_SONAR              = 0
HEADER_BATHY              = 2
HEADER_ATTITUDE           = 3
HEADER_Q_SINGLEBEAM       = 26
HEADER_Q_MULTITX          = 27
HEADER_Q_MULTIBEAM        = 28
HEADER_NAVIGATION         = 42
HEADER_7125               = 61
HEADER_POS_RAW_NAVIGATION = 107

HEADER_NAMES = {0: "sonar", 1: "notes", 2: "bathy", 3: "attitude", 4: "forward look", 5: "ELAC", 6: "raw serial",
                7: "embedded header", 8: "hidden sonar", 9: "Seaview processed bathy", 10: "Seaview depths",
                11: "high speed sensor", 12: "echo strength", 13: "georec", 14: "Klein raw bathy",
                15: "high speed sensor 2", 16: "ELAC XSE", 17: "bathy XYZA", 18: "K5000 bathy IQ", 19: "bathy snippet",
                20: "GPS", 21: "GPS statistics", 22: "single beam", 23: "gyro", 24: "Trackpoint", 25: "multibeam",
                26: "QPS single beam", 27: "QPS multi-transducer", 28: "QPS multibeam", 42: "navigation", 50: "time",
                60: "Benthos CAATI SARA", 61: "Reson 7125", 62: "Reson 7125 snippet", 65: "QINSy R2Sonic bathy",
                66: "QINSy R2Sonic FTS", 68: "R2Sonic bathy", 69: "R2Sonic FTS", 70: "CODA Echoscope data",
                71: "CODA Echoscope config", 72: "CODA Echoscope image", 73: "Edgetech 4600", 75: "Klein bathy",
                78: "Reson 7018 water column", 79: "R2Sonic water column", 84: "gyro (source time)",
                100: "Reson position", 102: "bathy proc", 103: "attitude proc", 104: "single beam proc",
                105: "aux proc", 106: "Klein 3000 data page", 107: "POS raw navigation", 108: "Klein V4 data page",
                199: "custom", 200: "user defined"}
KNOWN_HEADER_TYPES = set(HEADER_NAMES) | {67, 76}

# Packets that start with an XTFPINGHEADER/XTFBATHHEADER (Table H, sections 3.1.11-3.1.16, appendices 1-3).
# The ping header navigation of bathymetry packets is preferred: in a file with a towed sidescan as well,
# sonar ping headers may carry the towfish position.
SONAR_PING_HEADERS = {0, 4, 8, 12, 60}
BATHY_PING_HEADERS = {2, 19, 27, 28, 61, 62, 65, 66, 68, 69, 75, 78, 79}
# Bathymetry whose payload is a vendor format not documented in the XTF document, reported when there are no pings
UNDECODED_BATHY = {2, 5, 14, 16, 17, 18, 22, 25, 26, 27, 65, 68, 75}

EPOCH = datetime.datetime(1970, 1, 1)
DAY_MICROSECONDS = 86400000000

def epoch_microseconds_from(year, month, day, hour, minute, second, microseconds):
	"""Calendar time to microseconds since 1970-01-01. Returns None when the fields are not a valid time."""
	try:
		t = datetime.datetime(year, month, day, hour, minute, second)
	except ValueError:
		return None
	return int((t - EPOCH).total_seconds()) * 1000000 + microseconds

def epoch_microseconds_from_7ktime(year, day, seconds, hours, minutes):
	"""7KTIME (7k DFD Table 3: year, day of year 1-366, seconds, hours, minutes) to microseconds since 1970-01-01."""
	if not (1 <= day <= 366 and hours <= 23 and minutes <= 59 and 0.0 <= seconds < 61.0):
		return None
	try:
		days = (datetime.date(year, 1, 1) - EPOCH.date()).days + day - 1
	except ValueError:
		return None
	return ((days * 24 + hours) * 60 + minutes) * 60000000 + int(round(seconds * 1e6))

def full_sound_velocity(soundVelocity):
	"""Table H SoundVelocity: "Isis uses 750 (one way), some XTF files use 1500" (appendix 2: "SoundVelocity (half)")."""
	if not soundVelocity > 0.0:
		return math.nan
	return 2.0 * soundVelocity if soundVelocity < 1000.0 else soundVelocity

def geographic(latitude_deg, longitude_deg):
	return abs(latitude_deg) <= 90.0 and abs(longitude_deg) <= 180.0

class XtfReader(BaseReader):
	def __init__(self):
		super().__init__()
		self.include_extra_detections = False   #Reson 7027 multi-detect: lower priority detections in the same beam
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.file_header = {}                   #parsed XTFFILEHEADER, e.g. file_header["NavUnits"]
		self.channels = []                      #parsed CHANINFO structures

	def get_offset(self):
		"""Bathymetry sonar position re. vessel reference point (VCS: x fwd, y stbd, z down), from the first bathymetry
		CHANINFO. XTF offsets have X to starboard, Y forward and Z down (Table D), so (x, y, z) = (Y, X, Z)."""
		for channel in self.channels:
			if channel["TypeOfChannel"] == TYPE_BATHYMETRY:
				self.offset = np.array([channel["OffsetY"], channel["OffsetX"], channel["OffsetZ"]])
		return self.offset

	def position_lever_arm(self):
		"""Navigation system position re. vessel reference point (VCS), from the file header NavOffset. Raw navigation
		packets (types 42 and 107) are the positions of that navigation system."""
		h = self.file_header
		if not h:
			return np.zeros(3)
		return np.array([h["NavOffsetY"], h["NavOffsetX"], h["NavOffsetZ"]])

	def _read_file_header(self, buffer):
		"""XTFFILEHEADER and its CHANINFO structures (Tables C and D). Returns the offset of the first packet."""
		if len(buffer) < FILE_HEADER_BLOCK:
			raise ValueError("File shorter than an XTF file header")
		(FileFormat, SystemType, RecordingProgramName, RecordingProgramVersion, SonarName, SonarType, NoteString, ThisFileName,
		 NavUnits, NumberOfSonarChannels, NumberOfBathymetryChannels, NumberOfSnippetChannels, NumberOfForwardLookArrays,
		 NumberOfEchoStrengthChannels, NumberOfInterferometryChannels, Reserved1, Reserved2, ReferencePointHeight,
		 ProjectionType, SpheriodType, NavigationLatency, OriginY, OriginX, NavOffsetY, NavOffsetX, NavOffsetZ, NavOffsetYaw,
		 MRUOffsetY, MRUOffsetX, MRUOffsetZ, MRUOffsetYaw, MRUOffsetPitch, MRUOffsetRoll) = FILE_HEADER.unpack_from(buffer, 0)
		if FileFormat != FILE_FORMAT:
			raise ValueError(f"FileFormat is {FileFormat}, not {FILE_FORMAT}: not an XTF file")

		text = lambda b: b.split(b"\0")[0].decode("ascii", "replace")
		self.file_header = {
			"RecordingProgramName": text(RecordingProgramName), "RecordingProgramVersion": text(RecordingProgramVersion),
			"SonarName": text(SonarName), "SonarType": SonarType, "NoteString": text(NoteString), "ThisFileName": text(ThisFileName),
			"NavUnits": NavUnits, "NumberOfSonarChannels": NumberOfSonarChannels, "NumberOfBathymetryChannels": NumberOfBathymetryChannels,
			"ReferencePointHeight": ReferencePointHeight, "NavigationLatency": NavigationLatency,
			"NavOffsetY": NavOffsetY, "NavOffsetX": NavOffsetX, "NavOffsetZ": NavOffsetZ, "NavOffsetYaw": NavOffsetYaw,
			"MRUOffsetY": MRUOffsetY, "MRUOffsetX": MRUOffsetX, "MRUOffsetZ": MRUOffsetZ,
			"MRUOffsetYaw": MRUOffsetYaw, "MRUOffsetPitch": MRUOffsetPitch, "MRUOffsetRoll": MRUOffsetRoll}

		# Sonar CHANINFOs come first, then bathymetry (2.3). The header is 1024 bytes, growing by 1024 when needed
		numChannels = NumberOfSonarChannels + NumberOfBathymetryChannels
		header_size = max(FILE_HEADER_BLOCK, -(-(FILE_HEADER.size + numChannels * CHANINFO.size) // FILE_HEADER_BLOCK) * FILE_HEADER_BLOCK)
		self.channels = []
		for i in range(min(numChannels, (min(header_size, len(buffer)) - FILE_HEADER.size) // CHANINFO.size)):
			(TypeOfChannel, SubChannelNumber, CorrectionFlags, UniPolar, BytesPerSample, Reserved, ChannelName, VoltScale, Frequency,
			 HorizBeamAngle, TiltAngle, BeamWidth, OffsetX, OffsetY, OffsetZ, OffsetYaw, OffsetPitch, OffsetRoll, BeamsPerArray,
			 SampleFormat, ReservedArea2) = CHANINFO.unpack_from(buffer, FILE_HEADER.size + i * CHANINFO.size)
			self.channels.append({
				"TypeOfChannel": TypeOfChannel, "SubChannelNumber": SubChannelNumber, "ChannelName": text(ChannelName),
				"BytesPerSample": BytesPerSample, "Frequency": Frequency, "OffsetX": OffsetX, "OffsetY": OffsetY, "OffsetZ": OffsetZ,
				"OffsetYaw": OffsetYaw, "OffsetPitch": OffsetPitch, "OffsetRoll": OffsetRoll})

		# Some writers count channels differently: if no packet starts where expected, look at the next 1024 blocks
		for start in range(header_size, min(len(buffer), header_size + 16 * FILE_HEADER_BLOCK), FILE_HEADER_BLOCK):
			if buffer[start : start + 2] == MAGIC_BYTES:
				if start != header_size:
					sys.stderr.write(f"File header is {start} bytes, not the {header_size} expected from its channel counts\n")
				return start
		return header_size

	def read(self,filename):
		self.positions = []
		self.attitudes = []
		self.pings     = []

		# Repeated problems are reported once, with a count, at the end
		warnings = {}
		def warn(message):
			warnings[message] = warnings.get(message, 0) + 1

		attitude_packets = []		# (epoch_us, roll_rad, pitch_rad, heading_rad) from attitude packets with an absolute time
		attitude_tagged  = []		# (TimeTag_ms, roll_rad, pitch_rad, heading_rad) from attitude packets with only a time tag
		attitude_tags    = []		# (AttitudeTimeTag_ms, ping epoch_us) from ping headers, to time the tagged attitude
		navigation       = []		# (epoch_us, latitude_rad, longitude_rad, height_m) from type 42
		pos_raw          = []		# idem from type 107
		pos_raw_attitude = []		# (epoch_us, roll_rad, pitch_rad, heading_rad) from type 107
		header_positions = {True: {}, False: {}}	# per bathy/sonar ping headers: fix epoch_us -> (latitude_deg, longitude_deg)
		header_attitudes = {True: {}, False: {}}	# per bathy/sonar ping headers: ping epoch_us -> (roll, pitch, heading) in degrees
		ping_packets     = []		# bathymetry packets turned into Pings after the first pass
		undecoded_bathy  = {}		# header type -> count
		count_7006       = 0

		with open(filename, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
			file_size = len(buffer)
			offset = self._read_file_header(buffer)
			nav_latlong = self.file_header["NavUnits"] == NAV_UNITS_LATLONG

			def valid_packet_at(start):
				"""A packet that starts with the magic number and is followed by the end of file or another packet."""
				if start + PACKET_HEADER.size > file_size or buffer[start : start + 2] != MAGIC_BYTES:
					return False
				(numBytes,) = struct.unpack_from("<I", buffer, start + 10)
				end = start + numBytes
				return numBytes >= PACKET_HEADER.size and (end == file_size or (end + 2 <= file_size and buffer[end : end + 2] == MAGIC_BYTES))

			# Read file one packet at the time
			invalid_since = None	# start of a corrupted region while searching for the next packet
			while offset + PACKET_HEADER.size <= file_size:
				packet_start = offset

				# Read the start common to all packets
				(MagicNumber, HeaderType, SubChannelNumber, NumChansToFollow, Reserved1a, Reserved1b, NumBytesThisRecord) = PACKET_HEADER.unpack_from(buffer, packet_start)
				packet_end = packet_start + NumBytesThisRecord

				# The magic number "can be used to align the data stream to the start of a packet" (3.1.3)
				if MagicNumber != MAGIC_NUMBER or NumBytesThisRecord < PACKET_HEADER.size or packet_end > file_size:
					if invalid_since is None:
						invalid_since = packet_start
						if MagicNumber == MAGIC_NUMBER and packet_end > file_size:
							sys.stderr.write(f"Packet at offset {packet_start} ({NumBytesThisRecord} bytes) extends past the end of the file ({file_size - packet_start} bytes left)\n")
						else:
							sys.stderr.write(f"Invalid packet at offset {packet_start}, searching for the next magic number\n")
					candidate = buffer.find(MAGIC_BYTES, packet_start + 1)
					while candidate >= 0 and not valid_packet_at(candidate):
						candidate = buffer.find(MAGIC_BYTES, candidate + 1)
					if candidate < 0:
						break
					offset = candidate
					continue
				if invalid_since is not None:
					sys.stderr.write(f"Resynchronised at offset {packet_start}, skipped {packet_start - invalid_since} bytes\n")
					invalid_since = None

				offset = packet_end
				type_name = HEADER_NAMES.get(HeaderType, str(HeaderType))

				if HeaderType in SONAR_PING_HEADERS or HeaderType in BATHY_PING_HEADERS:
					#XTFPINGHEADER / XTFBATHHEADER	Sonar and bathymetry packets
					if NumBytesThisRecord < PING_HEADER.size:
						warn(f"Packet type {HeaderType} ({type_name}) shorter than a ping header, skipped")
						continue
					(MagicNumber, HeaderType, SubChannelNumber, NumChansToFollow, Reserved1a, Reserved1b, NumBytesThisRecord,
					 Year, Month, Day, Hour, Minute, Second, HSeconds, JulianDay, EventNumber, PingNumber, SoundVelocity, OceanTide, Reserved2,
					 ConductivityFreq, TemperatureFreq, PressureFreq, PressureTemp, Conductivity, WaterTemperature, Pressure, ComputedSoundVelocity,
					 MagX, MagY, MagZ, AuxVal1, AuxVal2, AuxVal3, AuxVal4, AuxVal5, AuxVal6, SpeedLog, Turbidity, ShipSpeed, ShipGyro,
					 ShipYcoordinate, ShipXcoordinate, ShipAltitude, ShipDepth, FixTimeHour, FixTimeMinute, FixTimeSecond, FixTimeHsecond,
					 SensorSpeed, KP, SensorYcoordinate, SensorXcoordinate, SonarStatus, RangeToFish, BearingToFish, CableOut,
					 Layback, CableTension, SensorDepth, SensorPrimaryAltitude, SensorAuxAltitude, SensorPitch, SensorRoll, SensorHeading, Heave, Yaw,
					 AttitudeTimeTag, DOT, NavFixMilliseconds, ComputerClockHour, ComputerClockMinute, ComputerClockSecond, ComputerClockHsec,
					 FishPositionDeltaX, FishPositionDeltaY, FishPositionErrorCode, OptionalOffset, CableOutHundredths, ReservedSpace2) = PING_HEADER.unpack_from(buffer, packet_start)

					# Ping time, in hundredths of seconds
					ping_epoch = epoch_microseconds_from(Year, Month, Day, Hour, Minute, Second, HSeconds * 10000)
					if ping_epoch is None:
						warn(f"Packet type {HeaderType} ({type_name}) with an invalid ping time, skipped")
						continue

					# Attitude packets without an absolute time are timed with this millisecond tag (Table H AttitudeTimeTag)
					if AttitudeTimeTag:
						attitude_tags.append((AttitudeTimeTag, ping_epoch))

					# Most recent navigation, timed by its fix time (already corrected for NavigationLatency)
					bathy = HeaderType in BATHY_PING_HEADERS
					latitude, longitude = (SensorYcoordinate, SensorXcoordinate) if (SensorYcoordinate or SensorXcoordinate) else (ShipYcoordinate, ShipXcoordinate)
					if latitude or longitude:
						fix_epoch = ping_epoch
						if FixTimeHour or FixTimeMinute or FixTimeSecond or FixTimeHsecond:
							fix_epoch = epoch_microseconds_from(Year, Month, Day, FixTimeHour, FixTimeMinute, FixTimeSecond, FixTimeHsecond * 10000)
							if fix_epoch is None:
								fix_epoch = ping_epoch
							elif fix_epoch - ping_epoch > DAY_MICROSECONDS // 2:
								fix_epoch -= DAY_MICROSECONDS	# fix before midnight, ping after
							elif ping_epoch - fix_epoch > DAY_MICROSECONDS // 2:
								fix_epoch += DAY_MICROSECONDS
						header_positions[bathy].setdefault(fix_epoch, (latitude, longitude))

					# Attitude "received closest in time to this sonar or bathymetry update"
					if SensorPitch or SensorRoll or SensorHeading:
						header_attitudes[bathy].setdefault(ping_epoch, (SensorRoll, SensorPitch, SensorHeading))

					if HeaderType == HEADER_Q_MULTIBEAM:
						#28	XTFBATHHEADER followed by NumChansToFollow XTFQPSMBEENTRY (3.1.16)
						entries_start = packet_start + PING_HEADER.size
						if entries_start + NumChansToFollow * Q_MULTIBEAM_ENTRY.size > packet_end:
							warn("QPS multibeam packet too short for its number of beams, skipped")
							continue
						ping_packets.append((HEADER_Q_MULTIBEAM, ping_epoch, full_sound_velocity(SoundVelocity), entries_start, NumChansToFollow))

					elif HeaderType == HEADER_7125:
						#61	Reson 7125 bathymetry: a 7k Data Record Frame and the 7027 or 7006 record (appendix 1)
						drf_start = packet_start + PING_HEADER.size
						if drf_start + DRF_7K.size > packet_end:
							warn("Reson 7125 packet too short for a 7k record frame, skipped")
							continue
						(ProtocolVersion, RthOffset, SyncPattern, RecordSize, OptionalDataOffset, OptionalDataIdentifier,
						 Year7k, Day7k, Seconds7k, Hours7k, Minutes7k, RecordVersion, RecordTypeIdentifier, DeviceIdentifier,
						 Reserved7k0, SystemEnumerator, Reserved7k1, Flags7k, Reserved7k2, Reserved7k3, TotalRecords, FragmentNumber) = DRF_7K.unpack_from(buffer, drf_start)
						if SyncPattern != SYNC_PATTERN_7K:
							warn("Reson 7125 packet without a valid 7k record frame, skipped")
							continue

						# "To distinguish the difference examine the DRF ... and note the value of the RecordTypeID field"
						if RecordTypeIdentifier == RECORD_7006:
							count_7006 += 1
							continue
						if RecordTypeIdentifier != RECORD_7027:
							warn(f"Reson 7125 packet with 7k record {RecordTypeIdentifier}, skipped")
							continue

						rth_start = drf_start + 4 + RthOffset	# the DRF offset counts from the sync pattern
						if rth_start + RAW_DETECTION_7027.size > packet_end:
							warn("Reson 7027 record too short, skipped")
							continue
						(SonarId, PingNumber7k, MultiPingSequence, NumDetections, DataFieldSize, DetectionAlgorithm, DetectionFlags, SamplingRate_Hz, TxAngle_rad, *Reserved7027) = RAW_DETECTION_7027.unpack_from(buffer, rth_start)
						if DataFieldSize < DETECTION_7027_MIN or rth_start + RAW_DETECTION_7027.size + NumDetections * DataFieldSize > packet_end or not SamplingRate_Hz > 0.0:
							warn("Reson 7027 record with invalid detection sizes or sampling rate, skipped")
							continue

						# The 7k time of the ping (microseconds) is more precise than the ping header time (hundredths)
						epoch_7k = epoch_microseconds_from_7ktime(Year7k, Day7k, Seconds7k, Hours7k, Minutes7k)
						ping_packets.append((HEADER_7125, epoch_7k if epoch_7k is not None else ping_epoch, full_sound_velocity(SoundVelocity),
						                     rth_start + RAW_DETECTION_7027.size, NumDetections, DataFieldSize, DetectionFlags, SamplingRate_Hz, TxAngle_rad))

					elif HeaderType in UNDECODED_BATHY:
						undecoded_bathy[HeaderType] = undecoded_bathy.get(HeaderType, 0) + 1

				elif HeaderType == HEADER_ATTITUDE:
					#3	XTFATTITUDEDATA
					if NumBytesThisRecord < ATTITUDE.size:
						warn("Attitude packet too short, skipped")
						continue
					(MagicNumber, HeaderType, SubChannelNumber, NumChansToFollow, Reserved1a, Reserved1b, NumBytesThisRecord, Reserved2a, Reserved2b,
					 EpochMicroseconds, SourceEpoch, Pitch, Roll, Heave, Yaw, TimeTag, Heading, Year, Month, Day, Hour, Minutes, Seconds, Milliseconds, Reserved3) = ATTITUDE.unpack_from(buffer, packet_start)

					# Positive roll is to starboard (port up) and positive pitch nose up, as in KMALL
					sample = (math.radians(Roll), math.radians(Pitch), math.radians(Heading))
					if SourceEpoch and EpochMicroseconds <= 999999:
						attitude_packets.append((SourceEpoch * 1000000 + EpochMicroseconds,) + sample)
						continue
					epoch_microseconds = epoch_microseconds_from(Year, Month, Day, Hour, Minutes, Seconds, Milliseconds * 1000) if Year and Milliseconds <= 999 else None
					if epoch_microseconds is not None:
						attitude_packets.append((epoch_microseconds,) + sample)
					elif TimeTag:
						attitude_tagged.append((TimeTag,) + sample)
					else:
						warn("Attitude packet without any time, skipped")

				elif HeaderType == HEADER_NAVIGATION:
					#42	XTFHEADERNAVIGATION, source time-stamped navigation
					if NumBytesThisRecord < NAVIGATION.size:
						warn("Navigation packet too short, skipped")
						continue
					(MagicNumber, HeaderType, Reserved, NumBytesThisRecord, Year, Month, Day, Hour, Minute, Second, Microseconds, SourceEpoch, TimeTag,
					 RawYcoordinate, RawXcoordinate, RawAltitude, TimeFlag, Reserved1) = NAVIGATION.unpack_from(buffer, packet_start)

					# Appendix 3: SourceEpoch and the microseconds must be set; otherwise fall back on the source date and time
					if Microseconds > 999999:
						epoch_microseconds = None
					elif SourceEpoch:
						epoch_microseconds = SourceEpoch * 1000000 + Microseconds
					else:
						epoch_microseconds = epoch_microseconds_from(Year, Month, Day, Hour, Minute, Second, Microseconds)
					if epoch_microseconds is None:
						warn("Navigation packet without a valid source time, skipped")
						continue

					# "Raw position from POSMV or other time stamped navigation source": taken as degrees when in range
					if not geographic(RawYcoordinate, RawXcoordinate):
						warn("Navigation packet with a position that is not lat/long, skipped")
						continue
					navigation.append((epoch_microseconds, math.radians(RawYcoordinate), math.radians(RawXcoordinate), RawAltitude))

				elif HeaderType == HEADER_POS_RAW_NAVIGATION:
					#107	XTFPOSRAWNAVIGATION
					if NumBytesThisRecord < POS_RAW_NAVIGATION.size:
						warn("POS raw navigation packet too short, skipped")
						continue
					(MagicNumber, HeaderType, SubChannelNumber, NumChansToFollow, Reserved1a, Reserved1b, NumBytesThisRecord,
					 Year, Month, Day, Hour, Minutes, Seconds, MicroSeconds, RawYcoordinate, RawXcoordinate, RawAltitude,
					 Pitch, Roll, Heave, Heading, Reserved2) = POS_RAW_NAVIGATION.unpack_from(buffer, packet_start)

					# MicroSeconds is actually 0-9999, tenths of milliseconds
					epoch_microseconds = epoch_microseconds_from(Year, Month, Day, Hour, Minutes, Seconds, MicroSeconds * 100) if MicroSeconds <= 9999 else None
					if epoch_microseconds is None:
						warn("POS raw navigation packet without a valid time, skipped")
						continue
					if geographic(RawYcoordinate, RawXcoordinate) and (RawYcoordinate or RawXcoordinate):
						pos_raw.append((epoch_microseconds, math.radians(RawYcoordinate), math.radians(RawXcoordinate), RawAltitude))
					if Pitch or Roll or Heading:
						pos_raw_attitude.append((epoch_microseconds, math.radians(Roll), math.radians(Pitch), math.radians(Heading)))

				elif HeaderType in KNOWN_HEADER_TYPES:
					# Known packets that are not used (yet)
					pass
				else:
					warn(f"Unknown packet type {HeaderType}")

			if invalid_since is not None:
				sys.stderr.write(f"No valid packet found after offset {invalid_since}, stopping\n")
			elif offset < file_size:
				sys.stderr.write(f"Truncated packet at offset {offset} ({file_size - offset} bytes left), stopping\n")

			# Bathymetry is read from the mapped file, so this happens before it is closed
			self._make_pings(buffer, ping_packets)

		if not self.pings:
			if count_7006:
				warn("Reson 7125 bathymetry is in the older 7006 record (needs the appended 7004 beam geometry), which is not read: no pings")
			if undecoded_bathy:
				names = ", ".join(f"{HEADER_NAMES[t]} ({t})" for t in sorted(undecoded_bathy))
				warn(f"Bathymetry packets whose payload is a vendor format not documented in the XTF format, not read: {names}")

		self._make_positions(navigation, pos_raw, header_positions, nav_latlong, warn)
		self._make_attitudes(attitude_packets, attitude_tagged, attitude_tags, pos_raw_attitude, header_attitudes, warn)

		for message, count in warnings.items():
			sys.stderr.write(f"{message}" + (f" ({count}x)" if count > 1 else "") + "\n")

	@staticmethod
	def _read_detection_7027(buffer, offset, size):
		"""One 7027 detection. Older, shorter versions are padded with zeros (like the C++ example in 7k DFD appendix E)."""
		if size >= DETECTION_7027.size:
			return DETECTION_7027.unpack_from(buffer, offset)
		return DETECTION_7027.unpack(bytes(buffer[offset : offset + size]) + bytes(DETECTION_7027.size - size))

	def _rx_angle_sign_7027(self, buffer, ping_packets):
		"""Beam zero is the first beam on the port side (7k DFD 2.5) and the 7k frame has X to starboard, so the rx angle
		should increase with the beam number. Check that against the data over the first and last detection of each ping."""
		votes = 0
		for packet in ping_packets:
			if packet[0] != HEADER_7125 or packet[4] < 2:
				continue
			(kind, epoch_microseconds, surface_sound_speed, data_start, numDetections, dataFieldSize, detectionFlags, samplingRate_Hz, txAngle_rad) = packet
			(firstBeam, _, firstRxAngle, *_) = self._read_detection_7027(buffer, data_start, dataFieldSize)
			(lastBeam, _, lastRxAngle, *_) = self._read_detection_7027(buffer, data_start + (numDetections - 1) * dataFieldSize, dataFieldSize)
			trend = (lastBeam - firstBeam) * (lastRxAngle - firstRxAngle)
			votes += (trend > 0) - (trend < 0)
		if votes < 0:
			sys.stderr.write("Reson rx angles decrease with the beam number (positive to port): flipping their sign so beam_angle is positive to starboard\n")
			return -1.0
		return 1.0

	def _make_pings(self, buffer, ping_packets):
		rx_angle_sign = self._rx_angle_sign_7027(buffer, ping_packets)

		for packet in ping_packets:
			if packet[0] == HEADER_Q_MULTIBEAM:
				(kind, ping_epoch, surface_sound_speed, entries_start, numBeams) = packet
				for i in range(numBeams):
					(Id, Intensity, Quality, TwoWayTravelTime, DeltaTime, BeamAngle, TiltAngle, Reserved0, Reserved1, Reserved2, Reserved3) = Q_MULTIBEAM_ENTRY.unpack_from(buffer, entries_start + i * Q_MULTIBEAM_ENTRY.size)
					if not TwoWayTravelTime > 0.0:
						continue	# no detection

					# The header time is the transmission time; DeltaTime gives the ping time per beam for profilers
					epoch_microseconds = ping_epoch + int(round(DeltaTime * 1e6))

					# The QPS quality has no documented scale, so it is not comparable to the KMALL quality factor
					quality = math.nan

					# "Beam Angle convention, Negative to port side, nadir beam 0degs, positive to starboard side", so no flip.
					# "Tilt angle convention positive forward"
					ping = Ping(epoch_microseconds,Id,quality,Intensity,surface_sound_speed,TwoWayTravelTime,math.radians(TiltAngle),math.radians(BeamAngle))
					self.pings.append(ping)
				continue

			(kind, epoch_microseconds, surface_sound_speed, data_start, numDetections, dataFieldSize, detectionFlags, samplingRate_Hz, txAngle_rad) = packet
			uncertainty_available = (detectionFlags & UNCERTAINTY_METHOD_MASK) != 0
			for i in range(numDetections):
				(beamDescriptor, detectionPoint, rxAngle_rad, flags, detectionQuality, uncertainty, signalStrength) = self._read_detection_7027(buffer, data_start + i * dataFieldSize, dataFieldSize)

				# Multi-detect gives up to five detections per beam; priority 0 is the main one
				if (flags >> DETECTION_PRIORITY_SHIFT) & DETECTION_PRIORITY_MASK and not self.include_extra_detections:
					continue

				# Fractional sample number with the zero sample at the transmit time (7k DFD appendix F)
				twoWayTravelTime_sec = detectionPoint / samplingRate_Hz

				# The uncertainty is an error normalised to the detection point; in % like the KMALL quality factor
				quality = 100.0 * uncertainty if uncertainty_available else math.nan

				ping = Ping(epoch_microseconds,beamDescriptor,quality,signalStrength,surface_sound_speed,twoWayTravelTime_sec,txAngle_rad,rx_angle_sign * rxAngle_rad)
				self.pings.append(ping)

	def _make_positions(self, navigation, pos_raw, header_positions, nav_latlong, warn):
		# One source only, so positions from two systems never get interleaved: source time-stamped navigation first,
		# then POS raw navigation, then the most recent fix copied into the ping headers
		samples = navigation or pos_raw
		if not samples:
			header = header_positions[True] or header_positions[False]
			if header and not nav_latlong:
				warn("Ping header navigation is in projected meters (NavUnits 0) and XTF does not record the projection: no positions")
			elif header:
				samples = [(epoch_microseconds, math.radians(latitude), math.radians(longitude), math.nan)
				           for epoch_microseconds, (latitude, longitude) in header.items() if geographic(latitude, longitude)]

		# Packets are written asynchronously (2.2), so sort
		for (epoch_microseconds, latitude_rad, longitude_rad, height_m) in sorted(samples):
			position = Position(epoch_microseconds, latitude_rad, longitude_rad, height_m)
			self.positions.append(position)

	def _make_attitudes(self, attitude_packets, attitude_tagged, attitude_tags, pos_raw_attitude, header_attitudes, warn):
		# Attitude packets with only a millisecond TimeTag are timed with the AttitudeTimeTag of the ping headers:
		# the offset between that clock and the ping times
		if attitude_tagged:
			if attitude_tags:
				offset = statistics.median(ping_epoch - tag * 1000 for (tag, ping_epoch) in attitude_tags)
				attitude_packets = attitude_packets + [(int(tag * 1000 + offset), roll, pitch, heading) for (tag, roll, pitch, heading) in attitude_tagged]
			else:
				warn("Attitude packets with only a time tag and no ping header AttitudeTimeTag to time them, skipped")

		# The high rate attitude packets first, then POS raw navigation, then the ping headers' attitude
		samples = attitude_packets or pos_raw_attitude
		if not samples:
			header = header_attitudes[True] or header_attitudes[False]
			samples = [(epoch_microseconds, math.radians(roll), math.radians(pitch), math.radians(heading))
			           for epoch_microseconds, (roll, pitch, heading) in header.items()]

		for (epoch_microseconds, roll_rad, pitch_rad, heading_rad) in sorted(samples):
			attitude = Attitude(epoch_microseconds, roll_rad, pitch_rad, heading_rad)
			self.attitudes.append(attitude)
