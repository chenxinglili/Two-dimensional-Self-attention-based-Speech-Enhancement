# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:21:45 2017

@author: Chenxing Li
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


f="./p257_250"

data=np.fromfile(f,dtype=np.float32, count=-1, sep="")
dim=257

data = np.reshape(data, (-1, dim)).transpose()

fig = plt.figure(figsize=(10,5)) # 10,5
ax = fig.add_subplot(1,1,1)
cax = ax.matshow(data, cmap="viridis", interpolation=None, origin="lower", aspect="auto")
fig.colorbar(cax)

ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel("Frame")
ax.set_ylabel("Frequency")
ax.set_title("Edge Mask")

plt.savefig("edge.pdf", format="pdf")
plt.show()