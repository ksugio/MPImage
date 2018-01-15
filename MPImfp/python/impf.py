#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MPImfp
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('sample.png', 0)
freq = np.zeros(1500, dtype=np.uint32)
seed = MPImfp.measure(img, 0, freq, 1000000, 12345, 1)

ave = np.sum(np.arange(freq.size)*np.array(freq, dtype=np.float)/np.sum(freq))
print 'Average =', ave

plt.plot(np.arange(freq.size), freq)
plt.xlabel('Pixel'), plt.ylabel('Frequency')
plt.show()
