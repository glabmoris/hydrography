import struct
import mmap
import sys
import math
import numpy as np

from mbes.BaseReader import BaseReader
from mbes.Ping import Ping
from mbes.Attitude import Attitude
from mbes.Position import Position

# "Unavailable" markers from EMdgmFormat.h
UNAVAILABLE_LATITUDE  = 200.0
UNAVAILABLE_LONGITUDE = 200.0

# Bit 0 of the sensor status fields: 1 = sensor is chosen as active
SENSOR_ACTIVE = 0x01

# MRZ sounding detectionType values
DETECTION_NORMAL   = 0
DETECTION_EXTRA    = 1
DETECTION_REJECTED = 2

class KmallReader(BaseReader):
	def __init__(self):
		self.use_realtime_approximation = False #The MRZ datagram contains a position and XYZ estimate to be used in real-time.
		self.include_extra_detections = False   #detectionType 1 (water-column targets etc.), not seabed
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}                  #parsed #IIP, e.g. installation["TRAI_RX1"]["X"]

	def lever_arm(self):
		"""Sonar position re. vessel reference point (VCS: x fwd, y stbd, z down), from #IIP.
		Along-track from the TX array, across-track and vertical from the RX array (Mills cross)."""
		tx = self.installation.get("TRAI_TX1")
		rx = self.installation.get("TRAI_RX1")
		if not tx or not rx:
			return np.zeros(3)
		return np.array([float(tx["X"]), float(rx["Y"]), float(rx["Z"])])

	def read(self,filename):
		self.positions = []
		self.attitudes = []
		self.pings     = []
		self.installation = {}

		with open(filename, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
			# Read file one datagram at the time
			datagram_header_format = "<I4sBBHII"
			datagram_header_size = struct.calcsize(datagram_header_format)
			offset = 0
			file_size = len(buffer)

			while offset + datagram_header_size <= file_size:
				datagram_start = offset

				# Read datagram header
				(datagram_size, datagram_type, datagram_version, system_id, echosounder_id, time_sec, time_nanosec) = struct.unpack_from(datagram_header_format, buffer,offset)

				# Every datagram is header + body + a trailing uint32 copy of its size
				if datagram_size < datagram_header_size + 4:
					raise ValueError(f"Invalid datagram size {datagram_size} bytes at offset {datagram_start}")
				if datagram_start + datagram_size > file_size:
					sys.stderr.write(f"Truncated datagram at offset {datagram_start} ({datagram_size} bytes, {file_size - datagram_start} left), stopping\n")
					break
				(trailing_size,) = struct.unpack_from("<I", buffer, datagram_start + datagram_size - 4)
				if trailing_size != datagram_size:
					raise ValueError(f"Datagram at offset {datagram_start}: size {datagram_size} != trailing size {trailing_size}")

				offset += datagram_header_size

				datagram_type = datagram_type.decode("ascii")

				if   datagram_type == "#IIP":
					#IIP	EMdgmIIP_def	Installation parameters and sensor setup.
					(numBytesCmnPart, info, status) = struct.unpack_from("<HHH", buffer, offset)
					text = bytes(buffer[offset + 6 : datagram_start + datagram_size - 4]).split(b"\0")[0].decode("ascii", "replace")
					for line in text.replace(",", "\n").splitlines():
						name, sep, values = line.partition(":")
						if sep and "=" in values:
							self.installation[name.strip()] = dict(kv.split("=", 1) for kv in values.split(";") if "=" in kv)
				elif datagram_type == "#IOP":
					#IOP	EMdgmIOP_def	Runtime parameters as chosen by operator.
					pass
				elif datagram_type == "#MRZ":
					#MRZ	EMdgmMRZ_def	Multibeam (M) raw range (R) and depth(Z) datagram
					ping_offset = offset

					# Partitions
					partition_format = "<HH"
					(numOfDgms, dgmNum) = struct.unpack_from(partition_format, buffer, offset)
					ping_offset += struct.calcsize(partition_format)

					#MBody
					(numBytesCmnPart, pingCnt, rxFansPerPing, rxFanIndex,swathsPerPing, swathAlongPosition, txTransducerInd, rxTransducerInd, numRxTransducers, algorithmType) = struct.unpack_from("<HHBBBBBBBB",buffer,ping_offset)
					ping_offset += numBytesCmnPart

					#Ping Info
					pinginfo_format = "<HHfBBBBBBH11fhhBBHIfffHHfHH6fBBBBddf"
					(numBytesInfoData, padding0, pingRate_Hz, beamSpacing, depthMode, subDepthMode, distanceBtwSwath, detectionMode, pulseForm, padding1, frequencyMode_Hz, freqRangeLowLim_Hz, freqRangeHighLim_Hz, maxTotalTxPulseLength_sec, maxEffTxPulseLength_sec, maxEffTxBandWidth_Hz, absCoeff_dBPerkm, portSectorEdge_deg, starbSectorEdge_deg, portMeanCov_deg, starbMeanCov_deg, portMeanCov_m, starbMeanCov_m, modeAndStabilisation, runtimeFilter1, runtimeFilter2, pipeTrackingStatus, transmitArraySizeUsed_deg, receiveArraySizeUsed_deg, transmitPower_dB, SLrampUpTimeRemaining, padding2, yawAngle_deg, numTxSectors, numBytesPerTxSector, headingVessel_deg, soundSpeedAtTxDepth_mPerSec, txTransducerDepth_m, z_waterLevelReRefPoint_m, x_kmallToall_m, y_kmallToall_m, latLongInfo, posSensorStatus, attitudeSensorStatus, padding3, latitude_deg, longitude_deg, ellipsoidHeightReRefPoint_m) = struct.unpack_from(pinginfo_format,buffer,ping_offset)

					#If the ping contains extended data (MRZ version >= 1), it follows the base struct
					pinginfo_ext_format = "<fBBH"
					base = struct.calcsize(pinginfo_format)
					if numBytesInfoData >= base + struct.calcsize(pinginfo_ext_format):
						(bsCorrectionOffset_dB,lambertsLawApplied,iceWindow,activeModes) = struct.unpack_from(pinginfo_ext_format, buffer, ping_offset + base)
					ping_offset += numBytesInfoData

					#TX Sectors
					sectors = {}
					txsector_format = "<BBBBfffffffBBH"
					txsector_ext_format = "<fff"
					base = struct.calcsize(txsector_format)

					for _ in range(numTxSectors):
						(txSectorNumb,txArrNumber,txSubArray,padding0,sectorTransmitDelay_sec,tiltAngleReTx_deg,txNominalSourceLevel_dB,txFocusRange_m,centreFreq_Hz,signalBandWidth_Hz,totalSignalLength_sec,pulseShading,signalWaveForm,padding1) = struct.unpack_from(txsector_format, buffer, ping_offset)
						if numBytesPerTxSector >= base + struct.calcsize(txsector_ext_format):
							(highVoltageLevel_dB, sectorTrackingCorr_dB, effectiveSignalLength_sec) = struct.unpack_from(txsector_ext_format,buffer,ping_offset + base)
						sectors[txSectorNumb] = [txArrNumber,txSubArray,padding0,sectorTransmitDelay_sec,tiltAngleReTx_deg,txNominalSourceLevel_dB,txFocusRange_m,centreFreq_Hz,signalBandWidth_Hz,totalSignalLength_sec,pulseShading,signalWaveForm,padding1]
						ping_offset += numBytesPerTxSector

					#RX Info
					(numBytesRxInfo,numSoundingsMaxMain,numSoundingsValidMain,numBytesPerSounding,WCSampleRate,seabedImageSampleRate,BSnormal_dB,BSoblique_dB,extraDetectionAlarmFlag,numExtraDetections,numExtraDetectionClasses,numBytesPerClass) = struct.unpack_from("<HHHHffffHHHH",buffer,ping_offset)
					ping_offset += numBytesRxInfo

					# Extra detection classes
					for _ in range(numExtraDetectionClasses):
						(numExtraDetInClass, padding, alarmFlag) = struct.unpack_from("<HbB", buffer, ping_offset)
						ping_offset += numBytesPerClass

					# Header time = time of first TX pulse in the ping
					epoch_microseconds = time_sec * 1000000 + (time_nanosec // 1000)

					# Soundings
					samples_count = 0
					for _ in range(numSoundingsMaxMain + numExtraDetections):
						(soundingIndex,txSectorNumb,detectionType,detectionMethod,rejectionInfo1,rejectionInfo2,postProcessingInfo,detectionClass,detectionConfidenceLevel,padding,rangeFactor,qualityFactor,detectionUncertaintyVer_m,detectionUncertaintyHor_m,detectionWindowLength_sec,echoLength_sec,WCBeamNumb,WCrange_samples,WCNomBeamAngleAcross_deg,meanAbsCoeff_dBPerkm,reflectivity1_dB,reflectivity2_dB,receiverSensitivityApplied_dB,sourceLevelApplied_dB,BScalibration_dB,TVG_dB,beamAngleReRx_deg,beamAngleCorrection_deg,twoWayTravelTime_sec,twoWayTravelTimeCorrection_sec,deltaLatitude_deg,deltaLongitude_deg,z_reRefPoint_m,y_reRefPoint_m,x_reRefPoint_m,beamIncAngleAdj_deg,realTimeCleanInfo,SIstartRange_samples,SIcentreSample,SInumSamples) = struct.unpack_from("<HBBBBBBBBH6fHH18f4H",buffer,ping_offset)
						samples_count += SInumSamples
						ping_offset += numBytesPerSounding

						# detectionType 2 = rejected: the range is interpolated/extrapolated from neighbours, not measured
						if detectionType == DETECTION_REJECTED:
							continue
						if detectionType == DETECTION_EXTRA and not self.include_extra_detections:
							continue

						if self.use_realtime_approximation:
							latitude  = latitude_deg  + deltaLatitude_deg
							longitude = longitude_deg + deltaLongitude_deg
							position = Position(epoch_microseconds,math.radians(latitude),math.radians(longitude),ellipsoidHeightReRefPoint_m)
							self.positions.append(position)

						# beamAngleReRx_deg is positive towards PORT (the port-most beam has the largest positive
						# angle and a negative y_reRefPoint_m). Ping.beam_angle follows the VCS (y = starboard),
						# so flip the sign here.
						intensity = 0 #TODO
						surface_sound_speed = soundSpeedAtTxDepth_mPerSec
						ping = Ping(epoch_microseconds,soundingIndex,qualityFactor,intensity,surface_sound_speed,twoWayTravelTime_sec,math.radians(sectors[txSectorNumb][4]),-math.radians(beamAngleReRx_deg))
						self.pings.append(ping)

					# Samples (seabed image), not used yet
					ping_offset += 2 * samples_count

				elif datagram_type == "#MWC":
					#MWC	EMdgmMWC_def	Multibeam (M) water (W) column (C) datagram
					pass
				elif datagram_type == "#SPO":
					#SPO	EMdgmSPO_def	Sensor (S) data for position (PO)
					(numBytesCmnPart, sensorSystem, sensorStatus, padding) = struct.unpack_from("<HHHH", buffer, offset)
					(timeFromSensor_sec, timeFromSensor_nanosec, posFixQuality_m, correctedLat_deg, correctedLong_deg, speedOverGround_mPerSec, courseOverGround_deg, ellipsoidHeightReRefPoint_m) = struct.unpack_from("<IIfddfff", buffer, offset + numBytesCmnPart)

					# Header time, not timeFromSensor: GGA carries no date, and timeFromSensor is not a usable epoch in practice
					epoch_microseconds = time_sec * 1000000 + (time_nanosec // 1000)

					# Corrected data is only provided for the active sensor; inactive ones carry the UNAVAILABLE markers
					active    = sensorStatus & SENSOR_ACTIVE
					available = correctedLat_deg != UNAVAILABLE_LATITUDE and correctedLong_deg != UNAVAILABLE_LONGITUDE

					if not self.use_realtime_approximation and active and available:
						position = Position(epoch_microseconds, math.radians(correctedLat_deg),math.radians(correctedLong_deg),ellipsoidHeightReRefPoint_m)
						self.positions.append(position)

				elif datagram_type == "#SKM":
					#SKM	EMdgmSKM_def	Sensor (S) KM binary sensor format
					skminfo_format = "<HBBHHHH"
					#SKMInfo
					(numBytesInfoPart, sensorSystem, sensorStatus, sensorInputFormat, numSamplesArray, numBytesPerSample, sensorDataContents) = struct.unpack_from(skminfo_format,buffer, offset)

					# Only use the active attitude sensor, otherwise two IMUs get interleaved
					if not (sensorStatus & SENSOR_ACTIVE):
						offset = datagram_start + datagram_size
						continue

					attitude_offset = offset + numBytesInfoPart
					for _ in range(numSamplesArray):
						(dgmType, numBytesDgm, dgmVersion, timeFromSensor_sec, timeFromSensor_nanosec, status, latitude_deg, longitude_deg, ellipsoidHeight_m, roll_deg, pitch_deg, heading_deg, heave_m, rollRate, pitchRate, yawRate, velNorth, velEast, velDown, latitudeError_m, longitudeError_m, ellipsoidHeightError_m, rollError_deg, pitchError_deg, headingError_deg, heaveError_m, northAcceleration, eastAcceleration, downAcceleration) = struct.unpack_from("<4sHHIII2d21f",buffer, attitude_offset)
						(delayedHeave_time_sec, delayedHeave_time_nanosec, delayedHeave_m) = struct.unpack_from("<IIf", buffer, attitude_offset + numBytesDgm)
						attitude_offset += numBytesPerSample

						# Each KMbinary sample carries its own timestamp; the SKM header time is only that of the first sample
						epoch_microseconds = timeFromSensor_sec * 1000000 + (timeFromSensor_nanosec // 1000)

						attitude = Attitude(epoch_microseconds, math.radians(roll_deg), math.radians(pitch_deg), math.radians(heading_deg))
						self.attitudes.append(attitude)

				elif datagram_type in ("#SVP", "#SVT", "#SCL", "#SDE", "#SHI", "#CPO", "#CHE", "#FCF", "#SPE", "#SPD", "#IBE", "#IBR", "#IBS"):
					# Known datagrams that are not used (yet)
					pass
				else:
					sys.stderr.write(f"Unknown datagram {datagram_type}\n")

				offset = datagram_start + datagram_size
