Compiling DASH
==============




DASH uses CMake to compile and requires a CUDA-capable GPU to run.  Currently DASH is set up for for Linux systems.

Requirements
^^^^^^^^^^^^

- CUDA 8.0 and GCC 4.7 - 5.4.x *or* CUDA 7.5 and GCC 4.7 - 4.9.x
- Boost with Python libraries
- CMake
- Python 2.7

Compiling
^^^^^^^^^

DASH can be compiled with the following commands


.. code-block:: bash
    
    #check out source code
    git checkout http://github.com/MICCoM/DASH-public .

    mkdir build

    cd build

    #Sets up Makefile.  If CMake cannot find any of the required 
    #libraries, you may have to manually specifiy their paths.
    #See CMake documentation for help.
    cmake ..

    make
    #or for faster compiling, make -j 4 to compile with 4 processors

This produces two files that you need: libDASH.so and DASH.so .
libDASH.so is the compiled simulation engine library.  The path to libDASH.so must be in your LD_LIBRARY_PATH environment variable to run DASH.  The second file, DASH.so, is the Python wrapper.  The path to DASH.so must be in your Python ``sys.path`` variable.  This is what lets Python find DASH when you write ``import DASH``.  Note that CMake will put DASH.so in different folders depending on your system configuration.  Find it using ``find . -name "DASH.so``.






