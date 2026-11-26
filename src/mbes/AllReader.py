import struct
import mmap
import sys
import math
import datetime
from types import SimpleNamespace
import numpy as np

from mbes.BaseReader import BaseReader
from mbes.Ping import Ping
from mbes.Attitude import Attitude
from mbes.Position import Position

# References are to the Kongsberg "EM datagram formats" instruction manual, 850-160692/W (January 2018).
# Every datagram is: number of bytes (4U, not counting itself), STX, type, EM model, date, time, counter,
# serial number, data, ETX and a checksum (2U, sum of the bytes between STX and ETX). Appendix E says the length
# is written in two bytes, but the introduction and every datagram table give 4U, which is what is used here.

STX = 0x02
ETX = 0x03
STX_BYTE = bytes([STX])
TRAILER_SIZE = 3			# ETX + checksum
MIN_NUM_BYTES = 16 + TRAILER_SIZE	# header after the length field, ETX and checksum

# Datagram types
DGM_RAW_RANGE_78       = ord("N")	# Raw range and angle 78: EM 122, EM 302, EM 710, ME70 BO, EM 2040, EM 2040C
DGM_RAW_RANGE_F        = ord("f")	# Raw range and beam angle (f): EM 120, EM 300, EM 1002, EM 2000, EM 3000, EM 3002
DGM_EXTRA_DETECTIONS   = ord("l")	# EM 2040 and EM 2040C with Slim Processing Unit
DGM_POSITION           = ord("P")
DGM_ATTITUDE           = ord("A")
DGM_NETWORK_ATTITUDE   = ord("n")
DGM_HEADING            = ord("H")
DGM_INSTALLATION_START = ord("I")
DGM_INSTALLATION       = {ord("I"), ord("i"), ord("r")}	# start, stop and remote information
DGM_DEPTH_ONLY         = {ord("D"), ord("X"), ord("F")}	# processed depths, or EM 3000 raw ranges (F) that need the depth datagram

PARSED_DATAGRAMS = {DGM_RAW_RANGE_78, DGM_RAW_RANGE_F, DGM_EXTRA_DETECTIONS, DGM_POSITION, DGM_ATTITUDE,
                    DGM_NETWORK_ATTITUDE, DGM_HEADING} | DGM_INSTALLATION

# Output datagrams in the manual that are not used (yet)
KNOWN_UNUSED_DATAGRAMS = {ord(c) for c in "DXKFSYkOChETGUWRJ301B"}

# Detection info (Table 29 note 3): bit 7 set = invalid detection (interpolated, estimated, rejected or no detection)
DETECTION_INVALID = 0x80

# Position system descriptor (Table 39 note 2): 10xx xxxx or 11xx xxxx = the position system is active
POSITION_SYSTEM_ACTIVE = 0x80
INVALID_4S = 0x7FFFFFFF		# "Invalid data are always identified by the highest positive number allowed in a field"

# Sensor system descriptor (Table 34 note 2, Table 35 note 1)
SENSOR_NUMBER_SHIFT   = 4
SENSOR_NUMBER_MASK    = 0x03
SENSOR_HEADING_ACTIVE = 0x01	# 1 = heading from the sensor is active
SENSOR_ROLL_INACTIVE  = 0x02	# 0 = roll from the sensor is active
SENSOR_PITCH_INACTIVE = 0x04	# 0 = pitch from the sensor is active
SENSOR_NOT_USED       = -1		# network attitude: function is not used

EPOCH_ORDINAL = datetime.date(1970, 1, 1).toordinal()

def epoch_microseconds_from(date, time_ms):
	"""Datagram date (year*10000 + month*100 + day) and time since midnight (ms) to microseconds since 1970-01-01.
	Returns None if invalid."""
	if time_ms > 86399999:
		return None
	try:
		days = datetime.date(date // 10000, date // 100 % 100, date % 100).toordinal() - EPOCH_ORDINAL
	except ValueError:
		return None
	return days * 86400000000 + time_ms * 1000

def byte_sum(buffer, start, end):
	"""Datagram checksum: sum of the bytes between STX and ETX, as a 16 bit unsigned number."""
	return int(np.frombuffer(buffer, dtype=np.uint8, count=end - start, offset=start).sum(dtype=np.uint64)) & 0xFFFF

def datagram_formats(byte_order):
	"""Datagram layouts in the byte order of the file. EM 3002, EM 710, EM 302, EM 122, EM 2040, EM 2040C and ME70BO
	are little endian; the manual recommends checking the byte order (output datagrams introduction)."""
	f = lambda fmt: struct.Struct(byte_order + fmt)
	return SimpleNamespace(
		length                 = f("I"),
		header                 = f("IBBHIIHH"),	# number of bytes, STX, type, EM model, date, time since midnight in ms, counter, serial number
		checksum               = f("H"),
		raw_range_78           = f("HHHHfI"),		# Table 29
		tx_sector_78           = f("hHfffHBBf"),
		rx_beam_78             = f("hBBHBbfhbB"),
		raw_range_f            = f("HHIiHHHH"),		# Table 28
		tx_sector_f            = f("hHIIIHBB"),
		rx_beam_f              = f("hHBbBBhH"),
		extra_detections       = f("HHHHHHfffHHHHHH"),	# Table 25
		extra_detection        = f("9fhbBHHHH"),	# first fields of a detection, the size per detection is in the datagram
		position               = f("iiHHHHBB"),		# Table 39
		attitude               = f("H"),			# Table 34
		attitude_entry         = f("HHhhhH"),
		network_attitude       = f("HbB"),			# Table 35
		network_attitude_entry = f("HhhhHB"),
		heading                = f("H"),			# Table 38
		heading_entry          = f("HH"),
		installation           = f("H"),			# Table 45, secondary system serial number
	)

def height_from_sentence(sentence):
	"""Antenna height above the ellipsoid from the position input datagram as received (Table 39 note 3, without '$'):
	GGA altitude + geoidal separation (as Kongsberg computes the height datagram, Table 1 note 2), or the GGK and
	PTNL,GGK ellipsoidal height (EHT). NaN for other formats, such as Simrad 90."""
	fields = sentence.decode("ascii", "replace").strip().lstrip("$").split("*")[0].split(",")
	try:
		if fields[0].endswith("GGA"):
			separation = float(fields[11]) if fields[11].strip() else 0.0
			return float(fields[9]) + separation
		if fields[0].endswith("GGK") or fields[0] == "PTNL":
			for field in fields:
				if field.startswith("EHT"):
					return float(field[3:])
	except (IndexError, ValueError):
		pass
	return math.nan

class AllReader(BaseReader):
	def __init__(self):
		super().__init__()
		self.include_extra_detections = False   #Extra detections datagram (water column targets etc.), not seabed
		self.verify_checksums = True            #Skip parsed datagrams whose checksum does not match
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}                  #parsed installation parameters, e.g. installation["S1X"]
		self.byte_order = "<"

	def _installation_xyz(self, prefix):
		p = self.installation
		return [float(p.get(prefix + axis, 0.0) or 0.0) for axis in "XYZ"]

	def lever_arm(self):
		"""Sonar position re. vessel reference point (VCS: x fwd, y stbd, z down), from the installation datagram.
		Along-track from the TX transducer, across-track and vertical from the RX transducer (Mills cross), as in
		KmallReader. Which of S0-S3 is which depends on the system transducer configuration (Table 45 notes 19 and 26).
		For sonar heads (EM 3002, EM 2040C/P/M) this is the centre of the head face; the arrays are offset from it
		(appendices B-D)."""
		if not self.installation:
			return np.zeros(3)
		stc = self.installation.get("STC")
		if stc is None:
			# Older datagrams have no STC: sonar head systems use S1 for (the first) head, others S1 = TX, S2 = RX
			heads = int(self.installation.get("emModel", 0)) in (2000, 3000, 3002, 3020, 2045)
			stc = 1 if heads else 0
		stc = int(float(stc))
		if stc in (1, 2, 5, 6, 9):
			return np.array(self._installation_xyz("S1"))
		tx = self._installation_xyz("S0" if stc == 4 else "S1")
		rx = self._installation_xyz("S2")
		return np.array([tx[0], rx[1], rx[2]])

	def position_lever_arm(self):
		"""Active position system (APS) location re. vessel reference point (VCS), from the installation datagram.
		Unlike KMALL #SPO positions, which are referred to the vessel reference point, .all positions are those of the
		positioning system: the XYZ 88 beam data are given re "the active positioning system's reference point"."""
		if not self.installation:
			return np.zeros(3)
		position_system = int(float(self.installation.get("APS", 0) or 0)) + 1	# APS is the position system number - 1
		return np.array(self._installation_xyz(f"P{position_system}"))

	@staticmethod
	def _detect_byte_order(buffer):
		"""Pick the byte order in which the first datagram has a consistent length, STX and ETX, and a plausible date."""
		candidates = []
		for byte_order in ("<", ">"):
			if len(buffer) < 4 + MIN_NUM_BYTES:
				break
			(numBytes, stx, datagramType, emModel, date, time_ms, counter, serialNumber) = struct.unpack_from(byte_order + "IBBHIIHH", buffer, 0)
			end = 4 + numBytes
			if stx == STX and numBytes >= MIN_NUM_BYTES and end <= len(buffer) and buffer[end - TRAILER_SIZE] == ETX:
				candidates.append((epoch_microseconds_from(date, time_ms) is not None, byte_order))
		if not candidates:
			raise ValueError("No valid datagram at the start of the file: not a Kongsberg .all file?")
		return max(candidates)[1]

	def read(self,filename):
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}

		# Repeated problems are reported once, with a count, at the end
		warnings = {}
		def warn(message):
			warnings[message] = warnings.get(message, 0) + 1

		attitude_streams = {}	# (datagram type, sensor number) -> [(epoch_us, roll_rad, pitch_rad, heading_rad, roll_pitch_active, heading_active)]
		headings = []			# (epoch_us, heading_rad) from active heading datagrams
		positions = []			# (epoch_us, latitude_rad, longitude_rad, height_m)
		ping_datagrams = []		# located in the first pass, turned into Pings in the second
		extra_detections = {}	# (serial number, ping counter) -> [(data start, ETX offset)]
		depth_only_count = 0

		with open(filename, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
			file_size = len(buffer)
			self.byte_order = self._detect_byte_order(buffer)
			fmt = datagram_formats(self.byte_order)

			def datagram_size(start):
				"""Size of a structurally valid datagram (including the length field) at start, else None."""
				if start + fmt.header.size > file_size:
					return None
				(numBytes,) = fmt.length.unpack_from(buffer, start)
				end = start + 4 + numBytes
				if numBytes < MIN_NUM_BYTES or end > file_size or buffer[start + 4] != STX or buffer[end - TRAILER_SIZE] != ETX:
					return None
				return 4 + numBytes

			def checksum_ok(start, size):
				(checksum,) = fmt.checksum.unpack_from(buffer, start + size - 2)
				return byte_sum(buffer, start + 5, start + size - TRAILER_SIZE) == checksum

			# Read file one datagram at the time
			offset = 0
			invalid_since = None	# start of a corrupted region while searching for the next valid datagram
			while offset + 4 + MIN_NUM_BYTES <= file_size:
				datagram_start = offset
				size = datagram_size(datagram_start)

				if size is None:
					# Resynchronise on the next STX that starts a datagram with a valid length, ETX and checksum
					if invalid_since is None:
						invalid_since = datagram_start
						(numBytes,) = fmt.length.unpack_from(buffer, datagram_start)
						if buffer[datagram_start + 4] == STX and datagram_start + 4 + numBytes > file_size:
							sys.stderr.write(f"Datagram at offset {datagram_start} ({4 + numBytes} bytes) extends past the end of the file ({file_size - datagram_start} bytes left)\n")
						else:
							sys.stderr.write(f"Invalid datagram at offset {datagram_start}, searching for the next valid one\n")
					candidate = datagram_start
					while True:
						next_stx = buffer.find(STX_BYTE, candidate + 5)
						if next_stx < 0:
							break
						candidate = next_stx - 4
						candidate_size = datagram_size(candidate)
						if candidate_size and checksum_ok(candidate, candidate_size):
							break
					if next_stx < 0:
						break
					offset = candidate
					continue
				if invalid_since is not None:
					sys.stderr.write(f"Resynchronised at offset {datagram_start}, skipped {datagram_start - invalid_since} bytes\n")
					invalid_since = None

				offset = datagram_start + size

				# Read datagram header
				(numBytes, stx, datagramType, emModel, date, time_ms, counter, serialNumber) = fmt.header.unpack_from(buffer, datagram_start)
				data_start = datagram_start + fmt.header.size
				etx_offset = datagram_start + size - TRAILER_SIZE
				type_name = chr(datagramType) if 32 <= datagramType < 127 else f"0x{datagramType:02X}"

				if datagramType in PARSED_DATAGRAMS:
					if self.verify_checksums and not checksum_ok(datagram_start, size):
						warn(f"Checksum mismatch, datagram {type_name} skipped")
						continue

				epoch_microseconds = epoch_microseconds_from(date, time_ms)
				if epoch_microseconds is None and datagramType in PARSED_DATAGRAMS:
					warn(f"Invalid date or time, datagram {type_name} skipped")
					continue

				if   datagramType in (DGM_RAW_RANGE_78, DGM_RAW_RANGE_F):
					#N	Raw range and angle 78, or f	Raw range and beam angle (older systems)
					ping_datagrams.append((datagramType, data_start, etx_offset, epoch_microseconds, serialNumber, counter))

				elif datagramType == DGM_EXTRA_DETECTIONS:
					#l	Extra detections. Its TX sector tilt is in the raw range and angle 78 datagram of the same ping
					if self.include_extra_detections:
						extra_detections.setdefault((serialNumber, counter), []).append((data_start, etx_offset))

				elif datagramType in DGM_INSTALLATION:
					#I, i, r	Installation parameters: ASCII "XXX=value," fields after the secondary serial number
					if self.installation and datagramType != DGM_INSTALLATION_START:
						continue	# keep the start datagram; the stop (i) or remote (r) one only if there is nothing else
					(secondarySerialNumber,) = fmt.installation.unpack_from(buffer, data_start)
					text = bytes(buffer[data_start + fmt.installation.size : etx_offset]).split(b"\0")[0].decode("ascii", "replace")
					installation = {"emModel": emModel, "systemSerialNumber": serialNumber, "secondarySystemSerialNumber": secondarySerialNumber}
					for field in text.split(","):
						name, sep, value = field.partition("=")
						if sep:
							installation[name.strip()] = value.strip()
					if datagramType == DGM_INSTALLATION_START or not self.installation:
						self.installation = installation

				elif datagramType == DGM_POSITION:
					#P	Position
					if data_start + fmt.position.size > etx_offset:
						warn("Datagram P too short, skipped")
						continue
					(latitude, longitude, fixQuality_cm, speedOverGround_cmPerSec, courseOverGround_cdeg, heading_cdeg, positionSystemDescriptor, numBytesInput) = fmt.position.unpack_from(buffer, data_start)

					# Only the active position system, otherwise two positioning systems get interleaved
					if not positionSystemDescriptor & POSITION_SYSTEM_ACTIVE:
						continue
					if latitude == INVALID_4S or longitude == INVALID_4S:
						continue
					latitude_deg  = latitude / 20000000.0
					longitude_deg = longitude / 10000000.0
					if abs(latitude_deg) > 90.0 or abs(longitude_deg) > 180.0:
						warn("Datagram P with an out of range position, skipped")
						continue

					# The P datagram has no height: take it from the position input datagram as received
					input_start = data_start + fmt.position.size
					height_m = height_from_sentence(bytes(buffer[input_start : min(input_start + numBytesInput, etx_offset)]))
					positions.append((epoch_microseconds, math.radians(latitude_deg), math.radians(longitude_deg), height_m))

				elif datagramType == DGM_ATTITUDE:
					#A	Attitude, valid at the transmit transducer. Entries are timed re the datagram time
					(numEntries,) = fmt.attitude.unpack_from(buffer, data_start)
					entries_start = data_start + fmt.attitude.size
					descriptor_offset = entries_start + numEntries * fmt.attitude_entry.size
					if descriptor_offset + 1 > etx_offset:
						warn("Datagram A too short for its number of entries, skipped")
						continue
					sensorSystemDescriptor = buffer[descriptor_offset]
					sensor = (sensorSystemDescriptor >> SENSOR_NUMBER_SHIFT) & SENSOR_NUMBER_MASK
					roll_pitch_active = not sensorSystemDescriptor & (SENSOR_ROLL_INACTIVE | SENSOR_PITCH_INACTIVE)
					heading_active = bool(sensorSystemDescriptor & SENSOR_HEADING_ACTIVE)

					samples = attitude_streams.setdefault((DGM_ATTITUDE, sensor), [])
					for i in range(numEntries):
						(time_since_start_ms, sensorStatus, roll_cdeg, pitch_cdeg, heave_cm, heading_cdeg) = fmt.attitude_entry.unpack_from(buffer, entries_start + i * fmt.attitude_entry.size)
						# Values outside the valid range are non-valid data (EM attitude input format)
						if abs(roll_cdeg) > 18000 or abs(pitch_cdeg) > 18000 or heading_cdeg > 35999:
							continue
						samples.append((epoch_microseconds + time_since_start_ms * 1000, math.radians(roll_cdeg * 0.01), math.radians(pitch_cdeg * 0.01), math.radians(heading_cdeg * 0.01), roll_pitch_active, heading_active))

				elif datagramType == DGM_NETWORK_ATTITUDE:
					#n	Network attitude velocity 110
					(numEntries, sensorSystemDescriptor, spare) = fmt.network_attitude.unpack_from(buffer, data_start)
					used = sensorSystemDescriptor != SENSOR_NOT_USED
					sensor = (sensorSystemDescriptor >> SENSOR_NUMBER_SHIFT) & SENSOR_NUMBER_MASK
					roll_pitch_active = used and not sensorSystemDescriptor & (SENSOR_ROLL_INACTIVE | SENSOR_PITCH_INACTIVE)
					heading_active = used and bool(sensorSystemDescriptor & SENSOR_HEADING_ACTIVE)

					samples = attitude_streams.setdefault((DGM_NETWORK_ATTITUDE, sensor), [])
					entry_offset = data_start + fmt.network_attitude.size
					for _ in range(numEntries):
						if entry_offset + fmt.network_attitude_entry.size > etx_offset:
							warn("Datagram n too short for its number of entries, truncated")
							break
						(time_since_start_ms, roll_cdeg, pitch_cdeg, heave_cm, heading_cdeg, numBytesInput) = fmt.network_attitude_entry.unpack_from(buffer, entry_offset)
						# Each entry carries the attitude input datagram as received
						entry_offset += fmt.network_attitude_entry.size + numBytesInput
						if abs(roll_cdeg) > 18000 or abs(pitch_cdeg) > 18000 or heading_cdeg > 35999:
							continue
						samples.append((epoch_microseconds + time_since_start_ms * 1000, math.radians(roll_cdeg * 0.01), math.radians(pitch_cdeg * 0.01), math.radians(heading_cdeg * 0.01), roll_pitch_active, heading_active))

				elif datagramType == DGM_HEADING:
					#H	Heading, for when the motion sensor does not supply the active heading
					(numEntries,) = fmt.heading.unpack_from(buffer, data_start)
					entries_start = data_start + fmt.heading.size
					indicator_offset = entries_start + numEntries * fmt.heading_entry.size
					if indicator_offset + 1 > etx_offset:
						warn("Datagram H too short for its number of entries, skipped")
						continue
					if buffer[indicator_offset] == 0:
						continue	# heading indicator: 0 = inactive
					for i in range(numEntries):
						(time_since_start_ms, heading_cdeg) = fmt.heading_entry.unpack_from(buffer, entries_start + i * fmt.heading_entry.size)
						if heading_cdeg <= 35999:
							headings.append((epoch_microseconds + time_since_start_ms * 1000, math.radians(heading_cdeg * 0.01)))

				elif datagramType in KNOWN_UNUSED_DATAGRAMS:
					# Known datagrams that are not used (yet)
					if datagramType in DGM_DEPTH_ONLY:
						depth_only_count += 1
				else:
					warn(f"Unknown datagram {type_name}")

			if invalid_since is not None:
				sys.stderr.write(f"No valid datagram found after offset {invalid_since}, stopping\n")
			elif offset < file_size:
				sys.stderr.write(f"Truncated datagram at offset {offset} ({file_size - offset} bytes left), stopping\n")

			# Datagrams are read from the mapped file, so this happens before it is closed
			self._make_pings(buffer, fmt, ping_datagrams, extra_detections, warn)

		if not ping_datagrams and depth_only_count:
			warn("No raw range and angle datagrams (N or f), only processed depths or EM 3000 F datagrams, which are not read: no pings")

		# Datagram timestamps are not strictly sequential, motion data for example arrives grouped (appendix E)
		for (epoch_microseconds, latitude_rad, longitude_rad, height_m) in sorted(positions):
			position = Position(epoch_microseconds, latitude_rad, longitude_rad, height_m)
			self.positions.append(position)

		self._make_attitudes(attitude_streams, headings, warn)

		for message, count in warnings.items():
			sys.stderr.write(f"{message}" + (f" ({count}x)" if count > 1 else "") + "\n")

	def _make_pings(self, buffer, fmt, ping_datagrams, extra_detections, warn):
		for (datagramType, data_start, etx_offset, epoch_microseconds, serialNumber, pingCounter) in ping_datagrams:
			if datagramType == DGM_RAW_RANGE_F:
				self._pings_from_raw_range_f(buffer, fmt, data_start, etx_offset, epoch_microseconds, warn)
				continue

			sectors = self._pings_from_raw_range_78(buffer, fmt, data_start, etx_offset, epoch_microseconds, warn)
			if sectors is None:
				continue

			# Extra detections follow the soundings of their ping, with sounding indices after them like in KMALL
			(tilts_rad, numRxBeams) = sectors
			for (extra_start, extra_etx) in extra_detections.pop((serialNumber, pingCounter), []):
				self._pings_from_extra_detections(buffer, fmt, extra_start, extra_etx, epoch_microseconds, tilts_rad, numRxBeams, warn)

		if extra_detections:
			warn("Extra detections without a raw range and angle 78 datagram for their ping, skipped")

	def _pings_from_raw_range_78(self, buffer, fmt, data_start, etx_offset, epoch_microseconds, warn):
		"""Raw range and angle 78 (Table 29). Returns the TX sector tilts and the number of RX beams, None if invalid."""
		(soundSpeed_dmPerSec, numTxSectors, numRxBeams, numValidDetections, samplingFrequency_Hz, dScale) = fmt.raw_range_78.unpack_from(buffer, data_start)
		tx_start = data_start + fmt.raw_range_78.size
		rx_start = tx_start + numTxSectors * fmt.tx_sector_78.size
		if rx_start + numRxBeams * fmt.rx_beam_78.size > etx_offset:
			warn("Datagram N too short for its number of sectors and beams, skipped")
			return None

		# TX sectors. The RX beams refer to a sector by its position in this loop: for the EM 2040 the
		# "Transmit sector number / TX array index" field here is the TX array index, not the sector (appendix A)
		tilts_rad = []
		for i in range(numTxSectors):
			(tiltAngle_cdeg, focusRange_dm, signalLength_sec, sectorTransmitDelay_sec, centreFrequency_Hz, meanAbsorptionCoeff_cdBPerkm, signalWaveformIdentifier, txArrayIndex, signalBandwidth_Hz) = fmt.tx_sector_78.unpack_from(buffer, tx_start + i * fmt.tx_sector_78.size)
			# Tilt angle re TX array, positive forwards (Table 27 note 1)
			tilts_rad.append(math.radians(tiltAngle_cdeg * 0.01))

		surface_sound_speed = soundSpeed_dmPerSec * 0.1

		# RX beams. All beams are included; the beam number is their index (output datagrams introduction)
		for beam in range(numRxBeams):
			(beamPointingAngle_cdeg, txSectorNumber, detectionInfo, detectionWindowLength, qualityFactor, dCorr, twoWayTravelTime_sec, reflectivity_ddB, realTimeCleaningInfo, spare) = fmt.rx_beam_78.unpack_from(buffer, rx_start + beam * fmt.rx_beam_78.size)

			# Bit 7 set = invalid: the range is interpolated, estimated, a rejected candidate or missing, not measured
			if detectionInfo & DETECTION_INVALID:
				continue
			if txSectorNumber >= numTxSectors:
				warn("Datagram N beam with a transmit sector number beyond its sectors, skipped")
				continue

			# Quality factor = 2500*sd/dr (note 2), so the standard deviation in % of the range, like the KMALL quality factor in %
			quality = qualityFactor / 25.0 if qualityFactor <= 254 else math.nan
			intensity = reflectivity_ddB * 0.1	# BS in dB

			# The beam pointing angle re RX array is positive to PORT (Table 27 note 1). Ping.beam_angle follows the VCS
			# (y = starboard), so flip the sign here, as in KmallReader.
			ping = Ping(epoch_microseconds,beam,quality,intensity,surface_sound_speed,twoWayTravelTime_sec,tilts_rad[txSectorNumber],-math.radians(beamPointingAngle_cdeg * 0.01))
			self.pings.append(ping)

		return (tilts_rad, numRxBeams)

	def _pings_from_raw_range_f(self, buffer, fmt, data_start, etx_offset, epoch_microseconds, warn):
		"""Raw range and beam angle (f) of older systems (Table 28). All beams in it are valid."""
		(numTxSectors, numBeams, samplingFrequency_cHz, rovDepth_cm, soundSpeed_dmPerSec, maxNumBeams, spare1, spare2) = fmt.raw_range_f.unpack_from(buffer, data_start)
		tx_start = data_start + fmt.raw_range_f.size
		rx_start = tx_start + numTxSectors * fmt.tx_sector_f.size
		if rx_start + numBeams * fmt.rx_beam_f.size > etx_offset:
			warn("Datagram f too short for its number of sectors and beams, skipped")
			return
		if samplingFrequency_cHz == 0:
			warn("Datagram f without a sampling frequency, skipped")
			return

		tilts_rad = {}
		for i in range(numTxSectors):
			(tiltAngle_cdeg, focusRange_dm, signalLength_us, transmitTimeOffset_us, centreFrequency_Hz, bandwidth_10Hz, signalWaveformIdentifier, txSectorNumber) = fmt.tx_sector_f.unpack_from(buffer, tx_start + i * fmt.tx_sector_f.size)
			tilts_rad[txSectorNumber] = math.radians(tiltAngle_cdeg * 0.01)

		surface_sound_speed = soundSpeed_dmPerSec * 0.1

		for i in range(numBeams):
			(beamPointingAngle_cdeg, range_quarterSamples, txSectorNumber, reflectivity_hdB, qualityFactor, detectionWindowLength, beamNumber, spare) = fmt.rx_beam_f.unpack_from(buffer, rx_start + i * fmt.rx_beam_f.size)
			if txSectorNumber not in tilts_rad:
				warn("Datagram f beam with an unknown transmit sector number, skipped")
				continue

			# Two way travel time = R / (4 * F / 100), with F the sampling frequency in 0.01 Hz (note 2)
			twoWayTravelTime_sec = range_quarterSamples / (4.0 * samplingFrequency_cHz / 100.0)

			# This quality factor is not defined as a standard deviation, so it is not comparable
			quality = math.nan
			intensity = reflectivity_hdB * 0.5	# BS in dB

			ping = Ping(epoch_microseconds,beamNumber,quality,intensity,surface_sound_speed,twoWayTravelTime_sec,tilts_rad[txSectorNumber],-math.radians(beamPointingAngle_cdeg * 0.01))
			self.pings.append(ping)

	def _pings_from_extra_detections(self, buffer, fmt, data_start, etx_offset, epoch_microseconds, tilts_rad, numRxBeams, warn):
		"""Extra detections (Table 25), timed and tilted like the raw range and angle 78 datagram of their ping."""
		(datagramCounter, datagramVersion, swathCounter, swathIndex, heading_cdeg, soundVelocity_dmPerSec, depthOfReferencePoint_m, wcSampleRate, siSampleRate, rxTransducerIndex, numExtraDetections, numDetectionClasses, numBytesPerClass, numAlarmFlags, numBytesPerDetection) = fmt.extra_detections.unpack_from(buffer, data_start)

		# Step by the sizes given in the datagram: Rev. T (June 2015) added data to the detections
		if numBytesPerDetection < fmt.extra_detection.size:
			warn(f"Datagram l with {numBytesPerDetection} bytes per detection, skipped")
			return
		detections_start = data_start + fmt.extra_detections.size + numDetectionClasses * numBytesPerClass
		if detections_start + numExtraDetections * numBytesPerDetection > etx_offset:
			warn("Datagram l too short for its number of detections, skipped")
			return

		surface_sound_speed = soundVelocity_dmPerSec * 0.1

		for i in range(numExtraDetections):
			(depth_m, across_m, along_m, deltaLatitude_deg, deltaLongitude_deg, beamPointingAngle_deg, appliedPointingAngleCorrection_deg, twoWayTravelTime_sec, appliedTwoWayTravelTimeCorrection_sec, bs_ddB, beamIncidenceAngleAdjustment_ddeg, detectionInfo, spare, txSectorNumber, detectionWindowLength, qualityFactorOld) = fmt.extra_detection.unpack_from(buffer, detections_start + i * numBytesPerDetection)
			if detectionInfo & DETECTION_INVALID:
				continue
			# Same TX sector index as in the raw range and angle 78 datagram (note 1)
			if txSectorNumber >= len(tilts_rad):
				warn("Datagram l detection with a transmit sector number beyond the ping's sectors, skipped")
				continue

			quality = qualityFactorOld / 25.0	# assumed scaled like the 78 quality factor
			intensity = bs_ddB * 0.1			# BS in dB

			ping = Ping(epoch_microseconds,numRxBeams + i,quality,intensity,surface_sound_speed,twoWayTravelTime_sec,tilts_rad[txSectorNumber],-math.radians(beamPointingAngle_deg))
			self.pings.append(ping)

	def _make_attitudes(self, attitude_streams, headings, warn):
		# Like KmallReader, only use the active attitude sensor, otherwise two sensors get interleaved. The sensor
		# system descriptor of each datagram tells whether its roll and pitch are used in real time.
		samples = [s for (datagramType, sensor), stream in attitude_streams.items() if datagramType == DGM_ATTITUDE for s in stream if s[4]]
		if not samples:
			samples = [s for (datagramType, sensor), stream in attitude_streams.items() if datagramType == DGM_NETWORK_ATTITUDE for s in stream if s[4]]
		if not samples and attitude_streams:
			samples = max(attitude_streams.values(), key=len)
			if samples:
				warn("No attitude datagram marked as the active roll/pitch sensor: using the sensor with the most samples")
		if not samples:
			return
		samples.sort()

		# The heading from the attitude sensor may not be the active one: then use the active heading, from another
		# attitude sensor or the heading datagram, interpolated to the attitude times
		heading_samples = sorted(headings + [(s[0], s[3]) for stream in attitude_streams.values() for s in stream if s[5]])
		need_heading = [i for i, s in enumerate(samples) if not s[5]]
		interpolated = {}
		if need_heading and heading_samples:
			heading_times = np.array([t for (t, h) in heading_samples], dtype=np.float64)
			heading_unwrapped = np.unwrap([h for (t, h) in heading_samples])
			times = np.array([samples[i][0] for i in need_heading], dtype=np.float64)
			values = np.mod(np.interp(times, heading_times, heading_unwrapped), 2.0 * math.pi)
			values[values >= 2.0 * math.pi] = 0.0	# np.mod can round a tiny negative value up to 2 pi
			interpolated = dict(zip(need_heading, values.tolist()))
		elif need_heading:
			warn("The attitude sensor's heading is not marked active and there is no other active heading: using it anyway")

		for i, (epoch_microseconds, roll_rad, pitch_rad, heading_rad, roll_pitch_active, heading_active) in enumerate(samples):
			attitude = Attitude(epoch_microseconds, roll_rad, pitch_rad, interpolated.get(i, heading_rad))
			self.attitudes.append(attitude)
