
Compiling on midway:
------------
Run the setupMidwayCompile.sh script
This replaces the standard CMakeLists.txt with a midway-specific one.

cd build
bash ../runCMake.sh

This should build two shared libraries - DASH.so, and libDASH.so
DASH.so is the python library.  Its directory must either be in your PYTHONPATH environment variable or added to the python variable sys.path on runtime
libDASH.so is the library which holds the compiled simulation code.  Its directory must be in your LD_LIBRARY_PATH environment varialble before running your simulation scripts.

CMake likes to hide these files.  Find using 
find . -name "libDASH.so" 
find . -name "DASH.so"


-----------
Compiling on a local machine:
-----------
mkdir build
cd build
cmake .. -DPYTHON=1
make  or make -j #PROCS (for faster build)

This should build two shared libraries - DASH.so, and libDASH.so
DASH.so is the python library.  Its directory must either be in your PYTHONPATH environment variable or added to the python variable sys.path on runtime
libDASH.so is the library which holds the compiled simulation code.  Its directory must be in your LD_LIBRARY_PATH environment varialble before running your simulation scripts.

CMake likes to hide these files.  Find using 
find . -name "libDASH.so" 
find . -name "DASH.so"


