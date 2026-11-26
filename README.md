# Hydrography library

Library to process echosounder data files into hydrographic data

# Usage

## Multibeam sonars
This module provides decoding and georeferencing of many binary file types recorded by sonars from a large variety manufacturers. It provides a unified hardware abstraction interface to handle hydrographic data processing tasks. 

### Examples
Support for KMALL,ALL,S7K and XTF binary file formats is provided through the MbesReader class.

```python
r = MbesReader()
pings = r.read(filename)

for p in pings:
	print(f"{p[0]} {p[1]} {p[2]}")
```
Pings are georeferenced in the ECEF terrestrial frame using the WGS84 ellipsoid.

Examples:

<img width="2318" height="773" alt="Screenshot from 2026-11-26 14-15-02" src="https://github.com/user-attachments/assets/d823e0a1-4610-4561-8d76-3e751009e1ed" />

KMALL files (Municipal Water Intake, Courtesy of CIDCO)


<img width="2325" height="774" alt="Screenshot from 2026-11-25 15-24-50" src="https://github.com/user-attachments/assets/0373b46c-8014-4e09-94fa-1aa0f2153d59" />
S7K files (Dredging Hole, Courtesy of CIDCO)


<img width="2325" height="774" alt="Screenshot from 2026-11-25 15-48-38" src="https://github.com/user-attachments/assets/e7cca89c-7fa4-402a-915b-6675d40b7a61" />
XTF files (Underwater Landslide, Courtesy of CIDCO)


# Developers

## Build
To build the package, run the `python -m build` command from the top level directory

## Testing
To test the package, run the following command from the top level directory:
`python3 -m unittest tests/test_core.py` 

Test data files are contained in the /tests/data/ folder
