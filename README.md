# Hydrography library

Library to process echosounder data files into hydrographic data

# Example Use

## Multibeam sonars
Support for KMALL,ALL, S7K and XTF formats is provided through the MbesReader class.

```python
  r = MbesReader()
	pings = r.read(sys.argv[1])

	for p in pings:
		print(f"{p[0]} {p[1]} {p[2]}")
```
Pings are georeferenced in the ECEF terrestrial frame using the WGS84 ellipsoid.

Examples:

<img width="2325" height="774" alt="Screenshot from 2026-11-25 15-24-50" src="https://github.com/user-attachments/assets/0373b46c-8014-4e09-94fa-1aa0f2153d59" />
S7K files (Courtesy of CIDCO)

<img width="2325" height="774" alt="Screenshot from 2026-11-25 15-48-38" src="https://github.com/user-attachments/assets/e7cca89c-7fa4-402a-915b-6675d40b7a61" />
XTF files (Courtesy of CIDCO)

# Developers

## Build
To build the package, run the `python -m build` command from the top level directory

## Testing
To test the package, run the following command from the top level directory:
`python3 -m unittest tests/test_core.py` 

Test data files are contained in the /tests/data/ folder
