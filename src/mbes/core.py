import os
import sys
import pandas as pd

from mbes.MbesReader import MbesReader

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.stderr.write("Usage: MbesReader.py sonar_file\n")
		sys.exit(1)

	r = MbesReader()
	pings = r.read(sys.argv[1])

	for p in pings:
		print(f"{p[0]} {p[1]} {p[2]}")

