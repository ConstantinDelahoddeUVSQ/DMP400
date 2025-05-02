import numpy as np
from scipy.constants import u, eV

m = 14 * u
v = np.sqrt(2e4 * eV / m )
print(v)
