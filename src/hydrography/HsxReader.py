import sys
import math
import datetime
from types import SimpleNamespace
import numpy as np

from hydrography.BaseReader import BaseReader
from hydrography.Ping import Ping
from hydrography.Attitude import Attitude
from hydrography.Position import Position

# References are to the Coastal Oceanographics technical note "HSX Format - Hysweep Text (ASCII) Logging", which
# describes HSX format version 7 (26-Jan-2010). An HSX file is text: a header section that ends with EOH, then a data
# section. Each record is one line starting with a three character tag. RMB (raw multibeam) and RSS (raw sidescan)
# records are followed by lines of values, one value per beam or sample, without a tag.
#
# Where the note leaves things open:
#  - RMB: the field list reads "dn t st sf bd n sv pn psa", but the examples are written "... bd sv pn n". HYPACK writes
#    the field list order (MB-System reads it that way too); the example order is recognised by the beam count.
#  - Time tags are seconds past midnight and the date is in TND. A backward jump of more than 12 hours is taken as a
#    midnight crossing. The note does not say whether the clock is UTC or local time; it is taken as UTC.
#  - POS records are grid coordinates (easting, northing) in work units, in a projection the file does not describe, so
#    the CRS has to be given. They carry no height.
#  - Data are logged as received from each device: POS is the antenna position, time tags include the device latency and
#    the MRU, gyro and sonar angles include their mounting angles. The OF2 offsets in the header are there for processing.

DAY_US   = 86400 * 1000000
HALF_DAY = 43200.0
EPOCH    = datetime.date(1970, 1, 1)

# HSP work units (p11): unit name as pyproj reports the axis unit of a CRS, and metres per work unit
WORK_UNITS = {0: ("metre", 1.0), 1: ("US survey foot", 1200.0 / 3937.0), 2: ("foot", 0.3048)}

# Sonar type codes (MBI st, RMB st)
SONAR_FIXED_ANGLES     = 1	# fixed beam roll angles (e.g. Reson Seabat): MBI first angle + beam number * increment
SONAR_VARIABLE_ANGLES  = 2	# variable beam roll angles (e.g. Seabeam SB1185), in the RMB record
SONAR_SPHERICAL        = 3	# beam info in spherical coordinates (e.g. Simrad EM3000)
SONAR_MULTI_TRANSDUCER = 4	# multiple transducers (e.g. Odom Miniscan): raw depths, one lever arm per transducer

# Sonar flags (MBI sf, RMB sf), hexadecimal
FLAG_ROLL_CORRECTED  = 0x0001	# beam angles are re. the vertical in roll: the sonar roll stabilises
FLAG_PITCH_CORRECTED = 0x0002	# same for pitch
FLAG_DUAL_HEAD       = 0x0004

# Beam data (MBI bd, RMB bd), hexadecimal. The RMB record is followed by one line per bit set, in bit order. Bits not
# listed (easting, northing, corrected depth, along/across distances, sounding flags, ...) are skipped.
BEAM_RANGES    = 0x0001	# slant ranges (multibeam) or raw depths (multiple transducer), work units
BEAM_PITCH     = 0x0040	# beam pitch angles, degrees, TSS convention
BEAM_ROLL      = 0x0080	# beam roll angles, degrees, TSS convention
BEAM_TAKEOFF   = 0x0100	# beam takeoff angles, degrees from vertical
BEAM_DIRECTION = 0x0200	# beam direction angles, degrees from forward
BEAM_DELAY     = 0x0400	# ping delay times, milliseconds
BEAM_INTENSITY = 0x0800
BEAM_QUALITY   = 0x1000	# quality codes from the sonar
USED_BEAM_DATA = {BEAM_RANGES, BEAM_PITCH, BEAM_ROLL, BEAM_TAKEOFF, BEAM_DIRECTION, BEAM_DELAY, BEAM_INTENSITY, BEAM_QUALITY}

# OF2 offset numbers
OFFSET_POSITION = 0	# position antenna
OFFSET_GYRO     = 1
OFFSET_MRU      = 2
OFFSET_HEAD_1   = 3	# sonar head 1 / transducer 1; head 2 / transducer 2 = 4, and so on

# Records in the note that are not used (yet). VER, COM and MSS are not in the note but later HYPACK versions write them.
KNOWN_UNUSED_RECORDS = {"FTP", "EOH", "EOL", "HVF", "LBP", "LIN", "LNN", "PTS", "SSI", "DFT", "FIX", "EC1", "GPS", "RSS",
                        "SNR", "TID", "VER", "COM", "MSS"}

def is_tag(token):
	"""Record tags are three upper case characters starting with a letter (RMB, OF2...). Value lines start with a digit
	or a sign, so this also tells the value lines that follow RMB and RSS records apart."""
	return len(token) == 3 and token[0].isalpha() and token.isupper()

def rotation(yaw, pitch, roll):
	"""Rz(yaw) Ry(pitch) Rx(roll), angles in radians. These are right-handed rotations about the VCS axes (x forward,
	y starboard, z down), so positive is clockwise seen from above, bow up and port side up: the OF2 sign conventions."""
	cy, sy = math.cos(yaw),   math.sin(yaw)
	cp, sp = math.cos(pitch), math.sin(pitch)
	cr, sr = math.cos(roll),  math.sin(roll)
	return np.array([
		[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
		[sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
		[-sp,   cp*sr,            cp*cr]
	])

def tss_vectors(roll, pitch):
	"""Unit vectors (N x 3, VCS) of beams given by TSS roll and pitch angles in radians. The sines of TSS angles are the
	direction cosines. Pitch, positive bow up, moves a beam forward. Roll, positive port side up, moves it to PORT, as
	rolling a vertical beam port side up would. NaN where the two angles do not describe a direction."""
	x = np.sin(pitch)
	y = -np.sin(roll)
	z2 = 1.0 - x*x - y*y
	z = np.sqrt(np.where(z2 >= 0.0, z2, np.nan))
	return np.column_stack((x, y, z))

class HsxReader(BaseReader):
	"""Reader for HYPACK/HYSWEEP HSX files.

	POS records are grid coordinates: give the CRS they are in, in anything pyproj accepts, in the file's work units:
	>>> reader = HsxReader(crs="EPSG:32619")
	>>> reader.read("0001_1200.HSX")
	"""

	def __init__(self, crs=None):
		super().__init__()
		self.crs = crs                          #CRS of the POS grid coordinates ("EPSG:32619", WKT, PROJ string...)
		self.ellipsoidal_height = 0.0           #POS records have no height: ellipsoidal height given to the antenna, m
		self.apply_latency = True               #subtract each device's OF2 latency from its time tags
		self.apply_mounting_angles = True       #take the OF2 rotation angles of the sonar head, MRU and gyro into account
		self.reject_low_quality = True          #skip beams whose quality code is below the HSP low quality limit (p8)
		self.sonar_head = 1                     #dual head sonars: the head whose beams are read (each has its own lever arm)
		self.position_device = None             #device numbers to read, None = automatic (PRI for positions, else the
		self.attitude_device = None             #lowest numbered enabled device with such records). HCP for roll and pitch,
		self.heading_device  = None             #GYR for heading,
		self.sonar_device    = None             #RMB for the soundings
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self._reset_header()

	def _reset_header(self):
		self.header    = {}                     #HSX version, TND, PRI, INF sound velocity, HSP fields
		self.devices   = {}                     #DEV and DV2 per device number
		self.multibeam = {}                     #MBI per device number
		self.offsets   = {}                     #OF2 per (device number, offset number), as written: work units, degrees, s
		self.sound_velocity_layers = []         #SVC (begin depth m, end depth m, sound velocity m/s)
		self.selected  = SimpleNamespace(position=None, attitude=None, heading=None, sonar=None)	#devices read
		self.metres_per_unit = 1.0

	# ------------------------------------------------------------------ #
	# Offsets
	# ------------------------------------------------------------------ #
	def _lever_arm(self, device, number):
		"""OF2 offset of a device re. the vessel reference point, VCS (x fwd, y stbd, z down), metres. Zeros if absent."""
		o = self.offsets.get((device, number))
		if o is None:
			return np.zeros(3)
		return np.array([o.forward, o.starboard, o.down]) * self.metres_per_unit

	def _latency_us(self, device, number):
		o = self.offsets.get((device, number))
		return round(o.latency * 1000000) if o is not None and self.apply_latency else 0

	def _mounting(self, device, number):
		"""OF2 rotation angles of a device as (yaw, pitch, roll) in radians, zeros if absent or not applied."""
		o = self.offsets.get((device, number))
		if o is None or not self.apply_mounting_angles:
			return (0.0, 0.0, 0.0)
		return (math.radians(o.yaw), math.radians(o.pitch), math.radians(o.roll))

	def lever_arm(self):
		"""Sonar head position re. vessel reference point (VCS: x fwd, y stbd, z down), from OF2."""
		return self._lever_arm(self.selected.sonar, OFFSET_HEAD_1 + self.sonar_head - 1)

	def position_lever_arm(self):
		"""Position antenna re. vessel reference point (VCS), from OF2."""
		return self._lever_arm(self.selected.position, OFFSET_POSITION)

	def get_offset(self):
		"""Sonar position re. the positions (VCS), which Georeferencing adds to them after rotating it by the attitude.
		POS records are antenna positions, so this is the sonar head offset minus the antenna offset."""
		self.offset = self.lever_arm() - self.position_lever_arm()
		return self.offset

	# ------------------------------------------------------------------ #
	# Reading
	# ------------------------------------------------------------------ #
	def read(self,filename):
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self._reset_header()

		# Repeated problems are reported once, with a count, at the end
		warnings = {}
		def warn(message):
			warnings[message] = warnings.get(message, 0) + 1

		positions = {}	# device -> [(epoch_us, easting, northing)]
		motions   = {}	# device -> [(epoch_us, heave_m, roll_deg, pitch_deg)]
		headings  = {}	# device -> [(epoch_us, heading_deg)]
		soundings = {}	# device -> [RMB records with their value lines]
		pitch_stabilisation = {}	# device -> (ping number, [head 1 angle, head 2 angle]) of the last PSA record
		layers = []		# SVC in work units

		# Time tags are seconds past midnight: the date comes from TND, the midnight crossings from the tags going back
		midnight_us = None
		last_tag = None
		def epoch_microseconds(tag):
			nonlocal midnight_us, last_tag
			if midnight_us is None:
				warn("No TND record before the data: times are seconds past midnight of 1970-01-01")
				midnight_us = 0
			if last_tag is not None:
				if tag > last_tag + HALF_DAY:
					return midnight_us - DAY_US + round(tag * 1000000)	# a late record from before midnight
				if tag < last_tag - HALF_DAY:
					midnight_us += DAY_US				# midnight has passed
			last_tag = tag
			return midnight_us + round(tag * 1000000)

		pending = None	# RMB record waiting for its value lines

		with open(filename, "r", encoding="latin-1") as file:
			for line in file:
				tokens = line.split()

				# Value lines of an RMB record: one per beam data bit, in bit order
				if pending is not None:
					if not tokens or not is_tag(tokens[0]):
						pending.lines.append(tokens)
						if len(pending.lines) == len(pending.bits):
							self._finish_rmb(pending, soundings, warn)
							pending = None
						continue
					warn("RMB record with fewer value lines than its beam data announces, skipped")
					pending = None

				if not tokens or not is_tag(tokens[0]):
					continue	# value lines of RSS and other records that are not read, blank lines

				tag = tokens[0]
				try:
					if   tag == "POS":
						#POS	Position: dn t x y. Grid easting and northing of the antenna, in work units
						positions.setdefault(int(tokens[1]), []).append((epoch_microseconds(float(tokens[2])), float(tokens[3]), float(tokens[4])))

					elif tag == "HCP":
						#HCP	Heave compensation: dn t heave(m) roll(deg, + port side up) pitch(deg, + bow up)
						motions.setdefault(int(tokens[1]), []).append((epoch_microseconds(float(tokens[2])), float(tokens[3]), float(tokens[4]), float(tokens[5])))

					elif tag == "GYR":
						#GYR	Gyro data: dn t heading(deg)
						headings.setdefault(int(tokens[1]), []).append((epoch_microseconds(float(tokens[2])), float(tokens[3])))

					elif tag == "RMB":
						#RMB	Raw multibeam data: dn t st sf bd n sv pn [psa], then one line of values per beam data bit
						pending = self._parse_rmb(tokens, epoch_microseconds, pitch_stabilisation)
						if not pending.bits:
							warn("RMB record without beam data, skipped")
							pending = None

					elif tag == "PSA":
						#PSA	Pitch stabilisation angle: dn t pn a0 a1, projector pitch angles of the ping in the next RMB
						pitch_stabilisation[int(tokens[1])] = (int(tokens[3]), [float(v) for v in tokens[4:6]])

					elif tag == "TND":
						#TND	Survey time and date: hh:mm:ss mm/dd/yy, when logging started
						if midnight_us is None:
							(hours, minutes, seconds) = (float(v) for v in tokens[1].split(":"))
							(month, day, year) = (int(v) for v in tokens[2].split("/"))
							if year < 100:
								year += 2000 if year < 70 else 1900
							midnight_us = (datetime.date(year, month, day) - EPOCH).days * DAY_US
							last_tag = hours * 3600.0 + minutes * 60.0 + seconds
							self.header["TND"] = datetime.datetime(year, month, day) + datetime.timedelta(seconds=last_tag)

					elif tag == "DEV":
						#DEV	Hypack device information: dn dc "name"
						device = self.devices.setdefault(int(tokens[1]), {})
						device["capabilities"] = int(tokens[2])
						device["name"] = line.split('"')[1] if line.count('"') >= 2 else " ".join(tokens[3:])

					elif tag == "DV2":
						#DV2	Hysweep device information: dn dc(hex) tf en
						device = self.devices.setdefault(int(tokens[1]), {})
						device["hysweep_capabilities"] = int(tokens[2], 16)
						device["towfish"] = tokens[3] == "1"
						device["enabled"] = tokens[4] == "1"

					elif tag == "OF2":
						#OF2	Hysweep device offsets: dn on starboard forward down(from waterline) yaw roll pitch latency
						values = [float(v) for v in tokens[3:10]]
						values += [0.0] * (7 - len(values))	# no latency in some files
						(starboard, forward, down, yaw, roll, pitch, latency) = values[:7]
						self.offsets[(int(tokens[1]), int(tokens[2]))] = SimpleNamespace(starboard=starboard, forward=forward, down=down, yaw=yaw, roll=roll, pitch=pitch, latency=latency)

					elif tag == "MBI":
						#MBI	Multibeam / multiple transducer device information: dn st sf bd n1 n2 fa ai, st sf bd in hex
						self.multibeam[int(tokens[1])] = SimpleNamespace(sonar_type=int(tokens[2], 16), sonar_flags=int(tokens[3], 16), beam_data=int(tokens[4], 16),
							beams_1=int(tokens[5]), beams_2=int(tokens[6]), first_angle=float(tokens[7]), angle_increment=float(tokens[8]))

					elif tag == "HSP":
						#HSP	Hysweep survey parameters: p1..p12. p8 = low beam quality (codes below are bad), p11 = work units
						p = tokens[1:]
						self.header["low_quality"] = float(p[7])
						if len(p) > 10:
							self.header["work_units"] = int(float(p[10]))
						if len(p) > 11:
							self.header["sonar_id"] = int(float(p[11]))

					elif tag == "INF":
						#INF	General information: "surveyor" "boat" "project" "area" tide draft sound_velocity
						self.header["sound_velocity"] = float(line.rsplit('"', 1)[-1].split()[2])

					elif tag == "SVC":
						#SVC	Sound velocity correction layer: begin depth, end depth (work units), sound velocity (m/s)
						layers.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))

					elif tag == "PRI":
						#PRI	Primary navigation device
						self.header["PRI"] = int(tokens[1])

					elif tag == "HSX":
						#HSX	HSX format version
						self.header["version"] = int(tokens[1])

					elif tag in KNOWN_UNUSED_RECORDS:
						# Known records that are not used (yet). RSS sample lines are skipped as value lines.
						pass
					else:
						warn(f"Unknown record {tag}")

				except (ValueError, IndexError):
					warn(f"Malformed {tag} record, skipped")

		if pending is not None:
			warn("RMB record truncated by the end of the file, skipped")

		# Work units, needed for ranges and offsets
		units = self.header.get("work_units")
		if units is None:
			warn("No work units (HSP record): metres assumed")
		elif units not in WORK_UNITS:
			warn(f"Unknown work units code {units} (HSP): metres assumed")
		self.metres_per_unit = WORK_UNITS.get(units, WORK_UNITS[0])[1]
		self.sound_velocity_layers = [(begin * self.metres_per_unit, end * self.metres_per_unit, sv) for (begin, end, sv) in layers]

		self._make_positions(positions, warn)
		self._make_attitudes(motions, headings, warn)
		self._make_pings(soundings, warn)	# after the attitudes: they may be needed to undo the sonar's stabilisation

		for message, count in warnings.items():
			sys.stderr.write(f"{message}" + (f" ({count}x)" if count > 1 else "") + "\n")

	@staticmethod
	def _parse_rmb(tokens, epoch_microseconds, pitch_stabilisation):
		"""RMB header. The beam count, sound velocity and ping number are kept in both orders (field list n sv pn, and
		the note's examples sv pn n) until the value lines tell which one has the right beam count."""
		device = int(tokens[1])
		layouts = []
		for (n, sv, pn) in ((tokens[6], tokens[7], tokens[8]), (tokens[8], tokens[6], tokens[7])):
			try:
				layouts.append((int(n), float(sv), int(pn)))
			except ValueError:
				pass
		if not layouts:
			raise ValueError("no beam count")
		beam_data = int(tokens[5], 16)
		return SimpleNamespace(
			device      = device,
			epoch_us    = epoch_microseconds(float(tokens[2])),
			sonar_type  = int(tokens[3], 16),
			sonar_flags = int(tokens[4], 16),
			beam_data   = beam_data,
			layouts     = layouts,
			psa         = pitch_stabilisation.pop(device, None),	# PSA records just precede the RMB of their ping
			bits        = [1 << k for k in range(16) if beam_data >> k & 1],
			lines       = [])

	@staticmethod
	def _finish_rmb(record, soundings, warn):
		counts = {len(tokens) for tokens in record.lines}
		if len(counts) != 1:
			warn("RMB value lines of different lengths, skipped")
			return
		count = counts.pop()
		layout = next((layout for layout in record.layouts if layout[0] == count), None)
		if layout is None:
			warn("RMB number of beams does not match its value lines, skipped")
			return
		(record.num_beams, record.sound_velocity, record.ping_number) = layout
		if record.psa is not None and record.ping_number != 0 and record.psa[0] not in (0, record.ping_number):
			record.psa = None
		# Values are converted when used: lines not read (sounding flags for example) need not be decimal numbers
		record.values = dict(zip(record.bits, record.lines))
		del record.lines, record.layouts
		soundings.setdefault(record.device, []).append(record)

	def _select(self, records, requested, attribute, preferred, warn):
		"""Device whose records are read: the requested one, else the first preferred one with records, else the lowest
		numbered enabled device with records. Like the active sensor in KmallReader: two devices of a kind would
		otherwise get interleaved."""
		candidates = sorted(device for device, r in records.items() if r)
		if requested is not None:
			if requested not in candidates:
				warn(f"No records from device {requested} ({attribute})")
				return None
			return requested
		if not candidates:
			return None
		enabled = [device for device in candidates if self.devices.get(device, {}).get("enabled", True)]
		choice = next((device for device in preferred if device in candidates), (enabled or candidates)[0])
		if len(candidates) > 1:
			warn(f"Records from devices {candidates}: device {choice} used, set {attribute} to choose another")
		return choice

	@staticmethod
	def _time_series(records, latency_us, *columns):
		"""Records sorted by time without repeated times (the interpolation needs distinct times): (times, columns...)."""
		records = sorted(records, key=lambda r: r[0])
		times = np.array([r[0] for r in records], dtype=np.int64) - latency_us
		times, first = np.unique(times, return_index=True)
		return (times,) + tuple(np.array([r[c] for r in records], dtype=np.float64)[first] for c in columns)

	def _make_positions(self, positions, warn):
		device = self._select(positions, self.position_device, "position_device", [self.header.get("PRI")], warn)
		self.selected.position = device
		if device is None:
			return
		if self.crs is None:
			warn("No CRS given (HsxReader(crs=...)): the POS grid coordinates cannot be converted to latitude and longitude, no positions")
			return

		import pyproj	# only needed here
		crs = pyproj.CRS.from_user_input(self.crs)
		units = self.header.get("work_units")
		if crs.is_projected and units in WORK_UNITS and crs.axis_info[0].unit_name != WORK_UNITS[units][0]:
			warn(f"The CRS is in {crs.axis_info[0].unit_name} but the POS coordinates are in work units ({WORK_UNITS[units][0]}, HSP)")
		to_geographic = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

		(times, eastings, northings) = self._time_series(positions[device], self._latency_us(device, OFFSET_POSITION), 1, 2)
		(longitudes, latitudes) = to_geographic.transform(eastings, northings)
		for (epoch_microseconds, latitude_deg, longitude_deg) in zip(times.tolist(), np.asarray(latitudes).tolist(), np.asarray(longitudes).tolist()):
			if not (math.isfinite(latitude_deg) and math.isfinite(longitude_deg)):
				warn("POS coordinates outside the CRS, skipped")
				continue
			position = Position(epoch_microseconds, math.radians(latitude_deg), math.radians(longitude_deg), self.ellipsoidal_height)
			self.positions.append(position)

	def _make_attitudes(self, motions, headings, warn):
		# Roll and pitch come from the MRU (HCP), heading from the gyro (GYR), each at its own times: attitudes are
		# made at the times of both, where both can be interpolated. Heave is not used, as in KmallReader.
		mru  = self._select(motions,  self.attitude_device, "attitude_device", [], warn)
		gyro = self._select(headings, self.heading_device,  "heading_device",  [], warn)
		self.selected.attitude, self.selected.heading = mru, gyro
		if mru is None or gyro is None:
			warn("Attitudes need HCP (roll, pitch) and GYR (heading) records: no attitudes")
			return

		(motion_times, roll_deg, pitch_deg) = self._time_series(motions[mru], self._latency_us(mru, OFFSET_MRU), 2, 3)
		(heading_times, heading_deg) = self._time_series(headings[gyro], self._latency_us(gyro, OFFSET_GYRO), 1)

		# The OF2 angles are the rotation of the device re. the vessel, so a device reads the vessel's angles plus its
		# own: take them off (small angles; the MRU yaw is not used)
		(mru_yaw, mru_pitch, mru_roll) = self._mounting(mru, OFFSET_MRU)
		(gyro_yaw, gyro_pitch, gyro_roll) = self._mounting(gyro, OFFSET_GYRO)
		roll    = np.radians(roll_deg) - mru_roll
		pitch   = np.radians(pitch_deg) - mru_pitch
		heading = np.unwrap(np.radians(heading_deg)) - gyro_yaw

		start = max(motion_times[0], heading_times[0])
		end   = min(motion_times[-1], heading_times[-1])
		times = np.union1d(motion_times, heading_times)
		times = times[(times >= start) & (times <= end)]
		if times.size == 0:
			warn("HCP and GYR records do not overlap in time: no attitudes")
			return

		roll    = np.interp(times, motion_times, roll)
		pitch   = np.interp(times, motion_times, pitch)
		heading = np.mod(np.interp(times, heading_times, heading), 2.0 * math.pi)
		heading[heading >= 2.0 * math.pi] = 0.0	# np.mod can round a tiny negative value up to 2 pi

		for (epoch_microseconds, roll_rad, pitch_rad, heading_rad) in zip(times.tolist(), roll.tolist(), pitch.tolist(), heading.tolist()):
			attitude = Attitude(epoch_microseconds, roll_rad, pitch_rad, heading_rad)
			self.attitudes.append(attitude)

	def _make_pings(self, soundings, warn):
		device = self._select(soundings, self.sonar_device, "sonar_device", sorted(self.multibeam), warn)
		self.selected.sonar = device
		if device is None:
			return

		mbi = self.multibeam.get(device)
		head = OFFSET_HEAD_1 + self.sonar_head - 1
		latency_us = self._latency_us(device, head)
		mounting = rotation(*self._mounting(device, head))
		low_quality = self.header.get("low_quality") if self.reject_low_quality else None
		default_sound_speed = self.header.get("sound_velocity") or 1500.0

		# Vessel roll and pitch, to take the sonar's own stabilisation out of its beam angles
		attitude_times = np.array([a.timestamp for a in self.attitudes], dtype=np.float64)
		attitude_roll  = np.array([a.roll for a in self.attitudes])
		attitude_pitch = np.array([a.pitch for a in self.attitudes])

		for record in soundings[device]:
			if record.sonar_type == SONAR_MULTI_TRANSDUCER:
				warn("Multiple transducer RMB records (sonar type 4) are not read: each transducer has its own lever arm")
				continue
			try:
				values = {bit: np.array(tokens, dtype=np.float64) for bit, tokens in record.values.items() if bit in USED_BEAM_DATA}
			except ValueError:
				warn("RMB value line that is not numeric, skipped")
				continue
			if BEAM_RANGES not in values:
				warn("RMB record without ranges, skipped")
				continue
			n = record.num_beams
			flags = record.sonar_flags
			epoch_microseconds = record.epoch_us - latency_us

			# A dual head sonar logs the n1 beams of head 1, then the n2 of head 2 (MBI)
			if flags & FLAG_DUAL_HEAD:
				if mbi is None or mbi.beams_1 + mbi.beams_2 != n:
					warn("Dual head RMB record whose number of beams is not n1 + n2 of the MBI record, skipped")
					continue
				beams = range(0, mbi.beams_1) if self.sonar_head == 1 else range(mbi.beams_1, n)
			elif self.sonar_head != 1:
				warn(f"sonar_head is {self.sonar_head} but the RMB record is not from a dual head sonar, skipped")
				continue
			else:
				beams = range(n)

			# Beam angles as TSS roll and pitch angles in degrees, re. the sonar head, or re. the vertical when the sonar
			# stabilises (flags)
			pitch_re_vertical = flags & FLAG_PITCH_CORRECTED
			if BEAM_TAKEOFF in values and BEAM_DIRECTION in values:
				# Takeoff from vertical, direction clockwise from forward
				takeoff = np.radians(values[BEAM_TAKEOFF])
				direction = np.radians(values[BEAM_DIRECTION])
				roll_deg  = -np.degrees(np.arcsin(np.sin(takeoff) * np.sin(direction)))
				pitch_deg = np.degrees(np.arcsin(np.sin(takeoff) * np.cos(direction)))
			else:
				if BEAM_ROLL in values:
					roll_deg = values[BEAM_ROLL]
				elif record.sonar_type == SONAR_FIXED_ANGLES and mbi is not None:
					roll_deg = mbi.first_angle + np.arange(n) * mbi.angle_increment
				else:
					warn("RMB record without beam angles (no roll or takeoff angles, and not a fixed angle sonar), skipped")
					continue
				if BEAM_PITCH in values:
					pitch_deg = values[BEAM_PITCH]
				elif record.psa is not None:
					# Pitch stabilised projector: its pitch angle, re. the head
					angles = record.psa[1]
					pitch_deg = np.full(n, angles[min(self.sonar_head - 1, len(angles) - 1)])
					pitch_re_vertical = False
				else:
					pitch_deg = np.zeros(n)

			# Stabilised angles are re. the vertical, but Georeferencing applies the full attitude: take the vessel's
			# roll and pitch at the ping time back out
			if flags & FLAG_ROLL_CORRECTED or pitch_re_vertical:
				if attitude_times.size < 2 or not attitude_times[0] <= epoch_microseconds <= attitude_times[-1]:
					warn("Stabilised beam angles but no attitude at the ping time to undo it, ping skipped")
					continue
				if flags & FLAG_ROLL_CORRECTED:
					roll_deg = roll_deg - math.degrees(np.interp(epoch_microseconds, attitude_times, attitude_roll))
				if pitch_re_vertical:
					pitch_deg = pitch_deg - math.degrees(np.interp(epoch_microseconds, attitude_times, attitude_pitch))

			# Into the vessel frame through the head's mounting angles. Ping angles: sin(tilt) and sin(beam angle) are the
			# forward and starboard direction cosines, as Georeferencing uses them
			vectors = tss_vectors(np.radians(roll_deg), np.radians(pitch_deg)) @ mounting.T
			tilts = np.arcsin(np.clip(vectors[:, 0], -1.0, 1.0))
			beam_angles = np.arcsin(np.clip(vectors[:, 1], -1.0, 1.0))
			downwards = vectors[:, 2] > 0.0		# False for NaN too

			# Slant ranges into two way travel times at the sound velocity Georeferencing will use with them
			ranges = values[BEAM_RANGES] * self.metres_per_unit
			sound_speed = record.sound_velocity if record.sound_velocity > 0 else default_sound_speed
			travel_times = 2.0 * ranges / sound_speed

			delays = values.get(BEAM_DELAY)
			intensities = values.get(BEAM_INTENSITY)
			qualities = values.get(BEAM_QUALITY)

			for beam in beams:
				# A range of 0 is no detection
				if not ranges[beam] > 0.0:
					continue
				if low_quality is not None and qualities is not None and qualities[beam] < low_quality:
					continue
				if not downwards[beam]:
					warn("Beams pointing upwards or with impossible angles, skipped")
					continue

				timestamp = epoch_microseconds + (round(delays[beam] * 1000) if delays is not None else 0)
				quality = float(qualities[beam]) if qualities is not None else math.nan
				intensity = float(intensities[beam]) if intensities is not None else 0
				ping = Ping(timestamp,beam,quality,intensity,sound_speed,float(travel_times[beam]),float(tilts[beam]),float(beam_angles[beam]))
				self.pings.append(ping)
