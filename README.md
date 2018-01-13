# MPImage
MPImage measures features of second phases in microstructures of materilas.

## Installation
Build MPImfp.pyd for Windows or MPImfp.so for Linux in MPImfp folder.
Build MPLn23d.pyd for Windows or MPLn23d.so for Linux in MPLn23d folder.

## References
+ METHODS
  + measure(img, barrier, f, nsample, seed, dflag) : measure image mean free path
    + img : input image
    + barrier : pixel value of barrier, 0 - 255
    + f : numpy dimension for result, dtype=np.uint32
    + nsample : number of sample
    + seed : seed for random number
    + dflag : measure mode, 0 : single mode, 1 : double mode
