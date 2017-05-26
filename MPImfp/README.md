# MPImfp
Python library to measure the image mean free path (IMFP).

## Compile in Windows
Visual Studio Community and Python library are required to create MPImfp.dll (MPImfp.pyd).
Open MPImfp.sln and set include path and library path according to [Property Manager] - [Microsoft.cpp.Win32.user] - [VC++ Directory] - [Include Directory] and [Library Directory].
Example of include path is shown.

    C:\Python27\include
    C:\Python27\Lib\site-packages\numpy\core\include

Example of library path is shown.

    C:\Python27\libs
    C:\Python27\Lib\site-packages\numpy\core\lib

Build in Release mode and MPImfp.pyd is copied to the root directory.

## Compile in Linux
Edit Makefile and execute make.

    vi Makefile
    make install

MPImfp.so is created and copied to the root directory.
