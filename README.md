# dmdpaper

This github repo is for reproducing the results in the DMD paper (Alan Kaptanoglu et al, 2019). Everything is in python and function/variable definitions are described in the doxygen file. To view the doxygen documentation, simply download the repo, cd into the directory, and type "doxygen Doxyfile". The html files can be opened with "google-chrome html/index.html". The .mat files for experimental shots 129175 and 129499, and the BIG-HIT simulation .mat files (with 8 IMPs or 32 IMPs) are included in this repository so that others can explicitly reproduce the results of this work. 

This code uses the "click" module for command line arguments and the "numba" module to parallelize some of the code. Much of the code is specific to HIT-SI format files, but the dmd.py file contains the methods and should be easily portable to some other format. To see a list of the command line options, type "python HITSI.py --help". A sample command line run would be:
"python HITSI.py --imp 8 --dmd 3 --dmd 2 --dmd 1 --dmd 4 --filenames bighit_14.mat --freqs 14.5 --limits 22.5 28.0 --directory ../HIT_data_analysis/dmd_files/ --nprocs 28 --trunc 30"
