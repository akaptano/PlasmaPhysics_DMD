# dmdpaper

This github repo is for reproducing the results in the DMD paper (Alan Kaptanoglu et al, 2019). Everything is in python and function/variable definitions are described in the doxygen file. To view the doxygen documentation, simply download the repo, cd into the directory, and type "doxygen Doxyfile". The html files can be viewed by opening a file like "html/index.html" in google chrome or safari. 

The .mat files for experimental shots 129175 and 129499, and the BIG-HIT simulation .mat files (with 8 IMPs or 32 IMPs) are included in this repository so that others can explicitly reproduce the results of this work. Adding these large files to the repository requires the use of the "GIT Large File Storage" extension, found here https://git-lfs.github.com/. 

This code uses the "click" module for command line arguments and the "numba" module to parallelize some of the code. Much of the code is specific to HIT-SI format files, but the dmd.py file contains the methods and should be easily portable to some other format. To see a list of the command line options, type "python HITSI.py --help". By default, the movies and pictures are written out to a directory called Pictures/ -- create this directory or change the default in order to avoid errors. 

A sample command line run would be:
"python HITSI.py --imp 8 --dmd 1 --dmd 3 --filenames bighit_8imp_14.mat --freqs 14.5 --limits 22.5 28.0 --directory /dmd_files/ --nprocs 28 --trunc 100"

