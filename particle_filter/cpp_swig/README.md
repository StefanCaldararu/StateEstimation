**HOW TO COMPILE SWIG CODE:** enter appropriate python environment where you want this module installed. Make sure swig is installed (apt-get, apt, brew, etc.) Then, run the following commands: 
swig -c++ -python particleFilter.i
python setup.py build_ext --inplace

Everything should work after this!!