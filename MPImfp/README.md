# MPImfp
Python library to measure the image mean free path (IMFP).

# Compile in Windows
Visual Studio Community and Python library are required to create MPImfp.dll (MPImfp.pyd).
Open MPImfp.sln and set include path and library path according to [Property Manager] - [Microsoft.cpp.Win32.user] - [VC++ Directory] - [Include Directory] and [Library Directory].
Example of include path is shown.

    C:\Python27\include
    C:\Python27\Lib\site-packages\numpy\core\include

Example of library path is shown.

    C:\Python27\libs
    C:\Python27\Lib\site-packages\numpy\core\lib

Build in Release mode and MPImfp.pyd is copied to the root directory.

# Compile in Linux
Edit Makefile and execute make.

    vi Makefile
    make install

MPImfp.so is created and copied to the root directory.

# Example

```python
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
```

# References
### Methods :
#### measure(img, barrier, f, nsample, seed, dflag)
measure image mean free path  
return seed
+ img : input image
+ barrier : pixel value of barrier, 0 - 255
+ f : numpy dimension for result, dtype=np.uint32
+ nsample : number of sample
+ seed : seed for random number
+ dflag : measure mode, 0 : single mode, 1 : double mode
