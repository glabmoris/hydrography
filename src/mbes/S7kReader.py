import struct
import mmap
import sys
import math
import datetime
import numpy as np

from mbes.BaseReader import BaseReader
from mbes.Ping import Ping
from mbes.Attitude import Attitude
from mbes.Position import Position

# References are to the Teledyne RESON "7k Data Format, Volume I", DFD 12713 version 2.41.
# All data is little endian with single byte alignment (2.6). Distances in metres, angles in radians (2.2).

# Data Record Frame (DRF), Table 5. The 7KTIME (Table 3) is inlined as HHfBB.
DRF          = struct.Struct("<HHIIIIHHfBBHIIHHIHHIII")	# 64 bytes
CHECKSUM     = struct.Struct("<I")
SYNC_PATTERN = 0x0000FFFF
SYNC_BYTES   = struct.pack("<I", SYNC_PATTERN)
SYNC_OFFSET  = 4			# the DRF "Offset" field (to the Record Type Header) counts from the sync pattern
PROTOCOL_VERSION = 5		# Table 4: only version 5 is in use
DRF_FLAG_CHECKSUM = 0x0001	# DRF flags bit 0: checksum is valid

# Record Type Headers (RTH) and Record Data (RD)
POSITION_1003         = struct.Struct("<IfdddBBBB")	# Table 12. A trailing "Number of satellites" (u8) is optional
ROLL_PITCH_HEAVE_1012 = struct.Struct("<fff")		# Table 25
HEADING_1013          = struct.Struct("<f")			# Table 26
NAVIGATION_1015       = struct.Struct("<Bdd6f")		# Table 29
ATTITUDE_1016         = struct.Struct("<B")			# Table 30
ATTITUDE_SAMPLE_1016  = struct.Struct("<H4f")		# Table 31
INSTALLATION_1020     = struct.Struct("<5I4f3f3f4f3f3ff3fI")	# Table 33, without the trailing 44 reserved bytes
SONAR_SETTINGS_7000   = struct.Struct("<QIH4fIIfHH5fII5fIfIIIfIf4f3fH")	# Table 36
RAW_DETECTION_7027    = struct.Struct("<QIHIIBIff16I")	# Table 66
DETECTION_7027        = struct.Struct("<HffIIff")		# Table 67, may be shorter (older) or longer (newer) in a file
DETECTION_7027_MIN    = struct.calcsize("<Hff")			# beam descriptor, detection point and rx angle are required
INSTALLATION_7030     = struct.Struct("<fH128sH128sH128sH128s6f6f6fH3fHf")	# Table 71

# Record type identifiers (Table 7)
RECORD_POSITION                 = 1003
RECORD_ROLL_PITCH_HEAVE         = 1012
RECORD_HEADING                  = 1013
RECORD_NAVIGATION               = 1015
RECORD_ATTITUDE                 = 1016
RECORD_INSTALLATION_IDENTIFIERS = 1020
RECORD_SONAR_SETTINGS           = 7000
RECORD_BATHYMETRIC_DATA         = 7006
RECORD_RAW_DETECTION_DATA       = 7027
RECORD_INSTALLATION_PARAMETERS  = 7030

PARSED_RECORDS = {RECORD_POSITION, RECORD_ROLL_PITCH_HEAVE, RECORD_HEADING, RECORD_NAVIGATION, RECORD_ATTITUDE,
                  RECORD_INSTALLATION_IDENTIFIERS, RECORD_SONAR_SETTINGS, RECORD_RAW_DETECTION_DATA, RECORD_INSTALLATION_PARAMETERS}
TIMED_RECORDS  = {RECORD_POSITION, RECORD_ROLL_PITCH_HEAVE, RECORD_HEADING, RECORD_NAVIGATION, RECORD_ATTITUDE, RECORD_RAW_DETECTION_DATA}

# Known records (Table 7, chapter 12) that are not used (yet)
KNOWN_UNUSED_RECORDS = (
	{1000, 1001, 1002, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1014, 1017, 1022, 1050, 1200, 2004, 5835, 5836}
	| set(range(1500, 1600))
	| {7001, 7002, 7003, 7004, 7005, 7007, 7008, 7009, 7010, 7011, 7012, 7016, 7017, 7018, 7019, 7021, 7022, 7023, 7024,
	   7026, 7028, 7031, 7037, 7038, 7040, 7041, 7042, 7048, 7050, 7051, 7052, 7053, 7055, 7057, 7058, 7059, 7060, 7068}
	| {7200, 7300, 7310, 7311, 7312} | set(range(7400, 7500))
	| {7500, 7501, 7502, 7503, 7504, 7505, 7510, 7511, 7515, 7610, 7611, 7612, 7613}
	| set(range(7900, 8000)) | {8012, 8100} | set(range(11000, 11300)) | set(range(81000, 88000))
)

# 1003 Position
DATUM_WGS84              = 0
POSITION_TYPE_GEOGRAPHIC = 0

# 7027 RTH flags bits 0-3: uncertainty method, 0 = not calculated
UNCERTAINTY_METHOD_MASK = 0x0F

# 7027 detection flags bits 9-12: detection priority within the same beam (multi-detect only), 0 = highest
DETECTION_PRIORITY_SHIFT = 9
DETECTION_PRIORITY_MASK  = 0x0F

EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()

def epoch_microseconds_from_7ktime(year, day, seconds, hours, minutes):
	"""7KTIME (Table 3: UTC year, day of year 1-366, seconds, hours, minutes) to microseconds since 1970-01-01.
	Returns None if no time is available (all fields zero, 2.7) or the fields are invalid."""
	if not (1 <= day <= 366 and hours <= 23 and minutes <= 59 and 0.0 <= seconds < 61.0):
		return None
	try:
		days = datetime.date(year, 1, 1).toordinal() - EPOCH_ORDINAL + day - 1
	except ValueError:
		return None
	return ((days * 24 + hours) * 60 + minutes) * 60000000 + int(round(seconds * 1e6))

def checksum(buffer, start, end):
	"""DRF checksum: unsigned 32-bit sum of all bytes from the protocol version field to the end of the data section."""
	return int(np.frombuffer(buffer, dtype=np.uint8, count=end - start, offset=start).sum(dtype=np.uint64)) & 0xFFFFFFFF

def largest_stream(streams):
	"""Samples of the (device identifier, system enumerator) stream with the most samples, so two sensors never get interleaved."""
	return max(streams.values(), key=len) if streams else []

class S7kReader(BaseReader):
	def __init__(self):
		super().__init__()
		self.include_extra_detections = False   #Multi-detect: lower priority detections in the same beam (priority > 0)
		self.verify_checksums = True            #Skip parsed records whose DRF checksum (when flagged as valid) does not match
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}                  #parsed 1020 and 7030, e.g. installation["7030"]["receiveArrayX_m"]

	def lever_arm(self):
		"""Sonar reference point re. vessel reference point, in the VCS used by KmallReader (x fwd, y stbd, z down).
		The 7k frame is X starboard, Y forward, Z up (2.2), so (x, y, z) = (Y, X, -Z).
		From 7030 (written by third party software such as Teledyne PDS): along-track from the TX array, across-track
		and vertical from the RX array, as the sonar reference point is defined (2.4). Otherwise the VRP to SRP offset from 1020."""
		parameters = self.installation.get("7030")
		if parameters:
			return np.array([parameters["transmitArrayY_m"], parameters["receiveArrayX_m"], -parameters["receiveArrayZ_m"]])
		identifiers = self.installation.get("1020")
		if identifiers:
			return np.array([identifiers["vrpToSrpY_m"], identifiers["vrpToSrpX_m"], -identifiers["vrpToSrpZ_m"]])
		return np.zeros(3)

	def read(self,filename):
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}

		# Repeated problems are reported once, with a count, at the end
		warnings = {}
		def warn(message):
			warnings[message] = warnings.get(message, 0) + 1

		# Sensor samples per stream, keyed by (device identifier, system enumerator)
		positions_1003  = {}	# (epoch_us, latitude_rad, longitude_rad, height_m)
		navigation_1015 = {}	# (epoch_us, latitude_rad, longitude_rad, height_m, heading_rad)
		rph_1012        = {}	# (epoch_us, roll_rad, pitch_rad)
		heading_1013    = {}	# (epoch_us, heading_rad)
		attitude_1016   = {}	# (epoch_us, roll_rad, pitch_rad, heading_rad)

		# 7027 carries no sound velocity: take it from the 7000 of the same ping, which may come before or after it
		sound_velocity_7000   = {}	# (device, enumerator, ping number, multi-ping sequence) -> sound velocity
		latest_sound_velocity = {}	# (device, enumerator) -> last 7000 sound velocity seen, fallback
		detections_7027 = []		# located in the first pass, turned into Pings in the second
		count_7006 = 0

		with open(filename, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
			file_size = len(buffer)
			offset = 0
			invalid_since = None	# start of a corrupted region while searching for the next sync pattern

			# Read file one record at the time
			while offset + DRF.size + CHECKSUM.size <= file_size:
				record_start = offset

				# Read Data Record Frame
				(protocolVersion, rthOffset, syncPattern, recordSize, optionalDataOffset, optionalDataIdentifier,
				 year, day, seconds, hours, minutes,
				 recordVersion, recordTypeIdentifier, deviceIdentifier, reserved0, systemEnumerator, reserved1,
				 flags, reserved2, reserved3, totalRecordsInFragmentedSet, fragmentNumber) = DRF.unpack_from(buffer, offset)

				record_end = record_start + recordSize
				rth_start  = record_start + SYNC_OFFSET + rthOffset
				body_end   = record_end - CHECKSUM.size

				# The sync pattern combined with the checksum allows recovery from corruption (chapter 6)
				valid_frame = (syncPattern == SYNC_PATTERN and recordSize >= DRF.size + CHECKSUM.size and
				               record_start + DRF.size <= rth_start <= body_end and record_end <= file_size)
				if not valid_frame:
					if invalid_since is None:
						invalid_since = record_start
						if syncPattern == SYNC_PATTERN and record_end > file_size:
							sys.stderr.write(f"Record at offset {record_start} ({recordSize} bytes) extends past the end of the file ({file_size - record_start} bytes left)\n")
						else:
							sys.stderr.write(f"Invalid record frame at offset {record_start}, searching for the next sync pattern\n")
					next_sync = buffer.find(SYNC_BYTES, record_start + SYNC_OFFSET + 1)
					if next_sync < 0:
						break
					offset = next_sync - SYNC_OFFSET
					continue
				if invalid_since is not None:
					sys.stderr.write(f"Resynchronised at offset {record_start}, skipped {record_start - invalid_since} bytes\n")
					invalid_since = None

				offset = record_end

				if protocolVersion != PROTOCOL_VERSION:
					warn(f"Protocol version {protocolVersion} (expected {PROTOCOL_VERSION}), parsed as version {PROTOCOL_VERSION}")

				# The optional data (if any) follows the record data
				if optionalDataOffset and rth_start <= record_start + optionalDataOffset <= body_end:
					body_end = record_start + optionalDataOffset
				body_size = body_end - rth_start

				if recordTypeIdentifier in PARSED_RECORDS:
					if self.verify_checksums and flags & DRF_FLAG_CHECKSUM:
						(recordChecksum,) = CHECKSUM.unpack_from(buffer, record_end - CHECKSUM.size)
						if checksum(buffer, record_start, record_end - CHECKSUM.size) != recordChecksum:
							warn(f"Checksum mismatch, record {recordTypeIdentifier} skipped")
							continue

				# Time at which the data was generated (chapter 9); for ping records when the transmitter finishes the ping
				epoch_microseconds = epoch_microseconds_from_7ktime(year, day, seconds, hours, minutes)
				if epoch_microseconds is None and recordTypeIdentifier in TIMED_RECORDS:
					warn(f"No valid time, record {recordTypeIdentifier} skipped")
					continue

				stream = (deviceIdentifier, systemEnumerator)

				if   recordTypeIdentifier == RECORD_POSITION:
					#1003	Position
					if body_size < POSITION_1003.size:
						warn("Record 1003 too short, skipped")
						continue
					(datumIdentifier, latency_sec, latitude_rad, longitude_rad, height_m, positionTypeFlag, utmZone, qualityFlag, positioningMethod) = POSITION_1003.unpack_from(buffer, rth_start)

					# Position only takes geographic WGS84 coordinates. The latency is not applied: positions use the time in the message (chapter 9)
					if positionTypeFlag != POSITION_TYPE_GEOGRAPHIC or datumIdentifier != DATUM_WGS84:
						warn("Record 1003 in grid coordinates or a non-WGS84 datum, skipped")
						continue
					positions_1003.setdefault(stream, []).append((epoch_microseconds, latitude_rad, longitude_rad, height_m))

				elif recordTypeIdentifier == RECORD_ROLL_PITCH_HEAVE:
					#1012	Roll Pitch Heave
					if body_size < ROLL_PITCH_HEAVE_1012.size:
						warn("Record 1012 too short, skipped")
						continue
					(roll_rad, pitch_rad, heave_m) = ROLL_PITCH_HEAVE_1012.unpack_from(buffer, rth_start)
					rph_1012.setdefault(stream, []).append((epoch_microseconds, roll_rad, pitch_rad))

				elif recordTypeIdentifier == RECORD_HEADING:
					#1013	Heading
					if body_size < HEADING_1013.size:
						warn("Record 1013 too short, skipped")
						continue
					(heading_rad,) = HEADING_1013.unpack_from(buffer, rth_start)
					heading_1013.setdefault(stream, []).append((epoch_microseconds, heading_rad))

				elif recordTypeIdentifier == RECORD_NAVIGATION:
					#1015	Navigation, at the vessel reference point
					if body_size < NAVIGATION_1015.size:
						warn("Record 1015 too short, skipped")
						continue
					(verticalReference, latitude_rad, longitude_rad, horizontalPositionAccuracy_m, vesselHeight_m, heightAccuracy_m, speedOverGround_mPerSec, courseOverGround_rad, heading_rad) = NAVIGATION_1015.unpack_from(buffer, rth_start)
					navigation_1015.setdefault(stream, []).append((epoch_microseconds, latitude_rad, longitude_rad, vesselHeight_m, heading_rad))

				elif recordTypeIdentifier == RECORD_ATTITUDE:
					#1016	Attitude, at the motion sensor rate
					if body_size < ATTITUDE_1016.size:
						warn("Record 1016 too short, skipped")
						continue
					(numberOfDataSets,) = ATTITUDE_1016.unpack_from(buffer, rth_start)
					if ATTITUDE_1016.size + numberOfDataSets * ATTITUDE_SAMPLE_1016.size > body_size:
						warn("Record 1016 too short for its number of data sets, skipped")
						continue

					samples = attitude_1016.setdefault(stream, [])
					attitude_offset = rth_start + ATTITUDE_1016.size
					for _ in range(numberOfDataSets):
						(timeDifference_ms, roll_rad, pitch_rad, heave_m, heading_rad) = ATTITUDE_SAMPLE_1016.unpack_from(buffer, attitude_offset)
						attitude_offset += ATTITUDE_SAMPLE_1016.size

						# Each data set is timed relative to the record timestamp
						samples.append((epoch_microseconds + timeDifference_ms * 1000, roll_rad, pitch_rad, heading_rad))

				elif recordTypeIdentifier == RECORD_INSTALLATION_IDENTIFIERS:
					#1020	Sonar Installation Identifiers
					if body_size < INSTALLATION_1020.size:
						warn("Record 1020 too short, skipped")
						continue
					(systemIdentificationNumber, transmitterIdentificationNumber, receiverIdentificationNumber, standardConfigurationOptions, configurationFixedParameters,
					 txLength_m, txWidth_m, txHeight_m, txRadius_m, srpToTxX_m, srpToTxY_m, srpToTxZ_m, txRoll_rad, txPitch_rad, txYaw_rad,
					 rxLength_m, rxWidth_m, rxHeight_m, rxRadius_m, srpToRxX_m, srpToRxY_m, srpToRxZ_m, rxRoll_rad, rxPitch_rad, rxYaw_rad,
					 frequency_Hz, vrpToSrpX_m, vrpToSrpY_m, vrpToSrpZ_m, cableLength_m) = INSTALLATION_1020.unpack_from(buffer, rth_start)
					self.installation["1020"] = {
						"srpToTxX_m": srpToTxX_m, "srpToTxY_m": srpToTxY_m, "srpToTxZ_m": srpToTxZ_m,
						"txRoll_rad": txRoll_rad, "txPitch_rad": txPitch_rad, "txYaw_rad": txYaw_rad,
						"srpToRxX_m": srpToRxX_m, "srpToRxY_m": srpToRxY_m, "srpToRxZ_m": srpToRxZ_m,
						"rxRoll_rad": rxRoll_rad, "rxPitch_rad": rxPitch_rad, "rxYaw_rad": rxYaw_rad,
						"vrpToSrpX_m": vrpToSrpX_m, "vrpToSrpY_m": vrpToSrpY_m, "vrpToSrpZ_m": vrpToSrpZ_m,
						"frequency_Hz": frequency_Hz}

				elif recordTypeIdentifier == RECORD_INSTALLATION_PARAMETERS:
					#7030	Sonar Installation Parameters, not generated by RESON sonars but by e.g. Teledyne PDS
					if body_size < INSTALLATION_7030.size:
						warn("Record 7030 too short, skipped")
						continue
					(frequency_Hz, firmwareVersionLength, firmwareVersion, softwareVersionLength, softwareVersion, software7kVersionLength, software7kVersion, recordProtocolLength, recordProtocolVersion,
					 transmitArrayX_m, transmitArrayY_m, transmitArrayZ_m, transmitArrayRoll_rad, transmitArrayPitch_rad, transmitArrayHeading_rad,
					 receiveArrayX_m, receiveArrayY_m, receiveArrayZ_m, receiveArrayRoll_rad, receiveArrayPitch_rad, receiveArrayHeading_rad,
					 motionSensorX_m, motionSensorY_m, motionSensorZ_m, motionSensorRollCalibration_rad, motionSensorPitchCalibration_rad, motionSensorHeadingCalibration_rad, motionSensorTimeDelay_sec,
					 positionSensorX_m, positionSensorY_m, positionSensorZ_m, positionSensorTimeDelay_sec, waterLineVerticalOffset_m) = INSTALLATION_7030.unpack_from(buffer, rth_start)
					self.installation["7030"] = {
						"transmitArrayX_m": transmitArrayX_m, "transmitArrayY_m": transmitArrayY_m, "transmitArrayZ_m": transmitArrayZ_m,
						"transmitArrayRoll_rad": transmitArrayRoll_rad, "transmitArrayPitch_rad": transmitArrayPitch_rad, "transmitArrayHeading_rad": transmitArrayHeading_rad,
						"receiveArrayX_m": receiveArrayX_m, "receiveArrayY_m": receiveArrayY_m, "receiveArrayZ_m": receiveArrayZ_m,
						"receiveArrayRoll_rad": receiveArrayRoll_rad, "receiveArrayPitch_rad": receiveArrayPitch_rad, "receiveArrayHeading_rad": receiveArrayHeading_rad,
						"motionSensorX_m": motionSensorX_m, "motionSensorY_m": motionSensorY_m, "motionSensorZ_m": motionSensorZ_m,
						"motionSensorRollCalibration_rad": motionSensorRollCalibration_rad, "motionSensorPitchCalibration_rad": motionSensorPitchCalibration_rad,
						"motionSensorHeadingCalibration_rad": motionSensorHeadingCalibration_rad, "motionSensorTimeDelay_sec": motionSensorTimeDelay_sec,
						"positionSensorX_m": positionSensorX_m, "positionSensorY_m": positionSensorY_m, "positionSensorZ_m": positionSensorZ_m,
						"positionSensorTimeDelay_sec": positionSensorTimeDelay_sec, "waterLineVerticalOffset_m": waterLineVerticalOffset_m,
						"frequency_Hz": frequency_Hz}

				elif recordTypeIdentifier == RECORD_SONAR_SETTINGS:
					#7000	7k Sonar Settings, updated for each ping
					if body_size < SONAR_SETTINGS_7000.size:
						warn("Record 7000 too short, skipped")
						continue
					(sonarId, pingNumber, multiPingSequence, frequency_Hz, sampleRate_Hz, receiverBandwidth_Hz, txPulseWidth_sec, txPulseTypeIdentifier, txPulseEnvelopeIdentifier, txPulseEnvelopeParameter, txPulseMode, txPulseReserved,
					 maxPingRate, pingPeriod_sec, rangeSelection_m, powerSelection_dB, gainSelection_dB, controlFlags,
					 projectorIdentifier, projectorBeamSteeringAngleVertical_rad, projectorBeamSteeringAngleHorizontal_rad, projectorBeamWidthVertical_rad, projectorBeamWidthHorizontal_rad, projectorBeamFocalPoint_m, projectorBeamWeightingWindowType, projectorBeamWeightingWindowParameter,
					 transmitFlags, hydrophoneIdentifier, receiveBeamWeightingWindow, receiveBeamWeightingParameter, receiveFlags, receiveBeamWidth_rad,
					 rangeFilterMin_m, rangeFilterMax_m, depthFilterMin_m, depthFilterMax_m, absorption_dBPerkm, soundVelocity_mPerSec, spreading_dB, reserved) = SONAR_SETTINGS_7000.unpack_from(buffer, rth_start)
					sound_velocity_7000[stream + (pingNumber, multiPingSequence)] = soundVelocity_mPerSec
					latest_sound_velocity[stream] = soundVelocity_mPerSec

				elif recordTypeIdentifier == RECORD_RAW_DETECTION_DATA:
					#7027	7k RAW Detection Data
					if body_size < RAW_DETECTION_7027.size:
						warn("Record 7027 too short, skipped")
						continue
					(sonarId, pingNumber, multiPingSequence, numDetections, dataFieldSize, detectionAlgorithm, detectionFlags, samplingRate_Hz, txAngle_rad, *reserved) = RAW_DETECTION_7027.unpack_from(buffer, rth_start)

					# The detection size is given by the RTH: older versions are shorter, newer ones have fields added at the end
					if dataFieldSize < DETECTION_7027_MIN:
						warn(f"Record 7027 with a detection size of {dataFieldSize} bytes, skipped")
						continue
					if RAW_DETECTION_7027.size + numDetections * dataFieldSize > body_size:
						warn("Record 7027 too short for its number of detections, skipped")
						continue
					if not samplingRate_Hz > 0.0:
						warn("Record 7027 without a valid sampling rate, skipped")
						continue

					detections_7027.append((epoch_microseconds, stream + (pingNumber, multiPingSequence), latest_sound_velocity.get(stream),
					                        rth_start + RAW_DETECTION_7027.size, numDetections, dataFieldSize, detectionFlags, samplingRate_Hz, txAngle_rad))

				elif recordTypeIdentifier == RECORD_BATHYMETRIC_DATA:
					#7006	7k Bathymetric Data, deprecated and superseded by 7027
					count_7006 += 1
				elif recordTypeIdentifier in KNOWN_UNUSED_RECORDS:
					# Known records that are not used (yet)
					pass
				else:
					warn(f"Unknown record type {recordTypeIdentifier}")

			if invalid_since is not None:
				sys.stderr.write(f"No valid record found after offset {invalid_since}, stopping\n")
			elif offset < file_size:
				sys.stderr.write(f"Truncated record at offset {offset} ({file_size - offset} bytes left), stopping\n")

			# Detections are read from the mapped file, so this happens before it is closed
			self._make_pings(buffer, detections_7027, sound_velocity_7000, warn)

		if not detections_7027 and count_7006:
			warn("The file only has the deprecated 7006 bathymetry (superseded by 7027), which is not read: no pings")

		self._make_positions(navigation_1015, positions_1003)
		self._make_attitudes(attitude_1016, rph_1012, heading_1013, navigation_1015, warn)

		for message, count in warnings.items():
			sys.stderr.write(f"{message}" + (f" ({count}x)" if count > 1 else "") + "\n")

	@staticmethod
	def _read_detection(buffer, offset, size):
		"""One 7027 detection. Older, shorter versions are padded with zeros (like the C++ example in appendix E)."""
		if size >= DETECTION_7027.size:
			return DETECTION_7027.unpack_from(buffer, offset)
		return DETECTION_7027.unpack(bytes(buffer[offset : offset + size]) + bytes(DETECTION_7027.size - size))

	def _rx_angle_sign(self, buffer, detections_7027):
		"""Beam zero is the first beam on the port side of the array (2.5) and the 7k frame has X to starboard (2.4), so the
		rx angle should increase with the beam number (figure 10-1). Ping.beam_angle is positive to starboard, like in
		KmallReader. Check that against the data rather than trust it: vote over the first and last detection of each ping."""
		votes = 0
		for (epoch_microseconds, ping_key, fallback_sound_velocity, data_offset, numDetections, dataFieldSize, detectionFlags, samplingRate_Hz, txAngle_rad) in detections_7027:
			if numDetections < 2:
				continue
			(firstBeam, _, firstRxAngle, *_) = self._read_detection(buffer, data_offset, dataFieldSize)
			(lastBeam, _, lastRxAngle, *_) = self._read_detection(buffer, data_offset + (numDetections - 1) * dataFieldSize, dataFieldSize)
			trend = (lastBeam - firstBeam) * (lastRxAngle - firstRxAngle)
			votes += (trend > 0) - (trend < 0)
		if votes < 0:
			sys.stderr.write("Rx angles decrease with the beam number (positive to port): flipping their sign so beam_angle is positive to starboard\n")
			return -1.0
		return 1.0

	def _make_pings(self, buffer, detections_7027, sound_velocity_7000, warn):
		rx_angle_sign = self._rx_angle_sign(buffer, detections_7027)

		for (epoch_microseconds, ping_key, fallback_sound_velocity, data_offset, numDetections, dataFieldSize, detectionFlags, samplingRate_Hz, txAngle_rad) in detections_7027:
			# Sound velocity at the sonar from the 7000 of this ping (appendix F), else the last one seen before it
			surface_sound_speed = sound_velocity_7000.get(ping_key, fallback_sound_velocity)
			if surface_sound_speed is None:
				warn("No 7000 sonar settings for a 7027 ping, surface sound speed set to NaN")
				surface_sound_speed = math.nan

			uncertainty_available = (detectionFlags & UNCERTAINTY_METHOD_MASK) != 0

			for i in range(numDetections):
				(beamDescriptor, detectionPoint, rxAngle_rad, flags, detectionQuality, uncertainty, signalStrength) = self._read_detection(buffer, data_offset + i * dataFieldSize, dataFieldSize)

				# Multi-detect gives up to five detections per beam; priority 0 is the main one
				priority = (flags >> DETECTION_PRIORITY_SHIFT) & DETECTION_PRIORITY_MASK
				if priority and not self.include_extra_detections:
					continue

				# The detection point is a fractional sample number with the zero sample at the transmit time (appendix F)
				twoWayTravelTime_sec = detectionPoint / samplingRate_Hz

				# The uncertainty is an error normalised to the detection point, i.e. a fraction of the range. Expressed in %
				# like the KMALL Ifremer quality factor. NaN when the sonar did not calculate it (RTH flags bits 0-3 = 0).
				quality = 100.0 * uncertainty if uncertainty_available else math.nan

				# Relative signal strength of the detection point, not calibrated
				intensity = signalStrength

				# Tx angle is the total applied steering, positive forward (2.2). The rx angle is relative to the receiver
				# in the sonar reference frame and includes the steering applied for roll stabilisation (appendix F).
				ping = Ping(epoch_microseconds,beamDescriptor,quality,intensity,surface_sound_speed,twoWayTravelTime_sec,txAngle_rad,rx_angle_sign * rxAngle_rad)
				self.pings.append(ping)

	def _make_positions(self, navigation_1015, positions_1003):
		# 1015 is explicitly the position of the vessel reference point, so it is preferred over 1003. Log files are not
		# guaranteed to be in chronological order (8.3), so samples are sorted.
		navigation = largest_stream(navigation_1015)
		if navigation:
			samples = [(epoch_microseconds, latitude_rad, longitude_rad, height_m) for (epoch_microseconds, latitude_rad, longitude_rad, height_m, heading_rad) in navigation]
		else:
			samples = largest_stream(positions_1003)

		for (epoch_microseconds, latitude_rad, longitude_rad, height_m) in sorted(samples):
			position = Position(epoch_microseconds, latitude_rad, longitude_rad, height_m)
			self.positions.append(position)

	def _make_attitudes(self, attitude_1016, rph_1012, heading_1013, navigation_1015, warn):
		# 1016 has complete, individually timed samples. Otherwise combine 1012 roll/pitch with the 1013 heading
		# (or the 1015 navigation heading), interpolated to the roll/pitch times.
		attitude = largest_stream(attitude_1016)
		if attitude:
			for (epoch_microseconds, roll_rad, pitch_rad, heading_rad) in sorted(attitude):
				self.attitudes.append(Attitude(epoch_microseconds, roll_rad, pitch_rad, heading_rad))
			return

		rph = sorted(largest_stream(rph_1012))
		if not rph:
			return

		headings = sorted(largest_stream(heading_1013))
		if not headings:
			headings = sorted((epoch_microseconds, heading_rad) for (epoch_microseconds, latitude_rad, longitude_rad, height_m, heading_rad) in largest_stream(navigation_1015))
		if not headings:
			warn("Roll and pitch (1012) without any heading (1013 or 1015): no attitudes")
			return

		heading_times = np.array([epoch_microseconds for (epoch_microseconds, heading_rad) in headings], dtype=np.float64)
		heading_unwrapped = np.unwrap([heading_rad for (epoch_microseconds, heading_rad) in headings])
		rph_times = np.array([epoch_microseconds for (epoch_microseconds, roll_rad, pitch_rad) in rph], dtype=np.float64)
		rph_headings = np.mod(np.interp(rph_times, heading_times, heading_unwrapped), 2.0 * math.pi)
		rph_headings[rph_headings >= 2.0 * math.pi] = 0.0	# np.mod can round a tiny negative value up to 2 pi

		for (epoch_microseconds, roll_rad, pitch_rad), heading_rad in zip(rph, rph_headings.tolist()):
			self.attitudes.append(Attitude(epoch_microseconds, roll_rad, pitch_rad, heading_rad))
