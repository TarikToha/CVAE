import numpy as np
from matplotlib import pyplot as plt

map_file = 'data/overall_azi_fft.v9.1_out/br_azi_fft_187_45.npy'
map_file = np.load(map_file)
print(map_file.shape, map_file.dtype)

plt.imshow(map_file, cmap='jet')
plt.colorbar()
plt.show()
