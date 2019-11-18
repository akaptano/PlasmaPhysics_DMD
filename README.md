# Introduction

This github repo is for reproducing the results in the DMD paper (Alan Kaptanoglu et al, 2019). Everything is in python and function/variable definitions are described in a doxygen file. To view the doxygen documentation, simply download the repo, cd into the directory, and type "doxygen Doxyfile" (requires the installation of doxygen on your machine). The html files can be viewed by opening a file like "html/index.html" in google chrome or safari. 

# Getting Started

Compatibility requires installation of Python 3.7, and the python packages scipy, numpy, matplotlib, click, and numba. These can be installed through pip or through an anaconda interface. 

This code uses the "click" module for command line arguments and the "numba" module to parallelize some of the code. Much of the code is specific to HIT-SI format files, but the dmd.py file contains the methods and should be easily portable to some other format. To see a list of the command line options, type "python HITSI.py --help". By default, the movies and pictures are written out to a directory called Pictures/. Create this directory or change the default in order to avoid errors. 

A sample command line run for analyzing BIG-HIT simulations with the bighit_dmd branch would be:
"python HITSI.py --imp 8 --dmd 1 --dmd 2 --filenames bighit_8imp_14.mat --freqs 14.5 --limits 22.7 28.5 --directory dmd_files/ --nprocs 28 --trunc 200"

and on the experiment_dmd branch, the analysis on an experimental shot (which has different file formatting and no internal probe arrays)
"python HITSI.py --imp 0 --filenames exppsi_129499.mat --freqs 14.5 --dmd 1 --dmd 3 --dmd 2 --trunc 30 --forecast True --directory /pylon5/ph5phhp/akaptano/dmd_files/ --limits 0.6 2.0"

# Reproducing Results

The .mat files for experimental shots 129175 and 129499, and the BIG-HIT simulation .mat files (with 8 IMPs or 32 IMPs) can be provided on request. Email akaptano@uw.edu for questions and requests. 

# License 

All files in this repository are available under the MIT License. Feel free to use and repurpose the code as you please. Variable projection routines in this repository were modified from matlab (https://github.com/duqbo/optdmd) to python, and parallelized. 

The MIT License (MIT)

Copyright (c) 2019 Alan Kaptanoglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
