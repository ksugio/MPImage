# MPLn23d
Python library to measure the 2-dimensional local number (LN2D) and the 3-dimensional local number (LN3D).

## Compile in Windows
Visual Studio Community and Python library are required to create MPLn23d.dll (MPLn23d.pyd).
Open MPLn23d.sln and set include path and library path according to [Property Manager] - [Microsoft.cpp.Win32.user] - [VC++ Directory] - [Include Directory] and [Library Directory].
Example of include path is shown.

    C:\Python27\include
    C:\Python27\Lib\site-packages\numpy\core\include

Example of library path is shown.

    C:\Python27\libs
    C:\Python27\Lib\site-packages\numpy\core\lib

Build in Release mode and MPLn23d.pyd is copied to the root directory.

## Compile in Linux
Edit Makefile and execute make.

    vi Makefile
    make install

MPLn23d.so is created and copied to the root directory.

## References
+ class ln2d_new(nsmax)
  + nsmax : maximum number of section
  + Methods:
    + add_gc(sid, x, y, r) : add gc (gravity center)
      1. sid : section id
      2. x, y : position of gc
      3. r : radius of gc
    + add_gc_random(sid, ngc, sd, r) : add gc randomly
      1. sid : section id
      2. ngc : number of gc
      3. sd : standard deviation of random number
      4. r: radius of gc
    + add_sec(step, sx, sy) : add section
    + area_fraction() : calculate area fraction
    + cut(step, ln3d, cid, dir, pos) : cut ln3d cell
    + cut_random(nsec, step, ln3d, cid) : cut ln3d cell randomly
    + measure_gc(f) : measure on gc
    + measure_random(f, nsample) : measure on random point  
  + Data descriptors:
    + nsec : number of section
    + nsec_max : maximum number of section
    + seed : seed of random number  
+ class ln3d_new(ncmax)
  + ncmax : maximum number of cell
  + Methods:
    + add_cell(step, sx, sy, sz) : add cell
    + add_gc(cid, x, y, z, r) : add gc
    + add_gc_random(cid, ngc, sd, r) : add gc random
    +  measure_gc(f) : measure on gc
    + measure_random(f, nsample) : measure on random point
    + volume_fraction() : calculate volume fraction
  + Data descriptors defined here:
    + ncell : number of cell
    + ncell_max : maximum number of cell
    + seed : seed of random number
