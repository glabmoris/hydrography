import numpy as np

class LeastSquares:
	@staticmethod
	def fit(points):
		"""Least-squares line through ordered XYZ points.
		Returns the start point and the total displacement vector (first -> last, along the line)."""
		pts = np.asarray(points, dtype=np.float64)
		centroid = pts.mean(axis=0)
		_, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
		direction = vt[0]                          # unit vector along the best-fit line

		t = (pts - centroid) @ direction           # each point's position along the line
		if t[-1] < t[0]:                           # orient it from the first point to the last
			direction, t = -direction, -t
		start = centroid + t[0] * direction
		end = centroid + t[-1] * direction
		return start, end - start
