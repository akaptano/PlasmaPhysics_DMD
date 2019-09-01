#!/usr/bin/env python
from plot_attributes import *
from psitet_load import loadshot
from psitet_utilities import SVD, \
    toroidal_modes_sp, poloidal_modes, \
    toroidal_modes_imp
from dmd import DMD_slide, DMD_forecast
from dmd_utilities import \
    make_reconstructions, \
    dmd_animation, \
    toroidal_plot
import click
#import cProfile, pstats, io
#from pstats import SortKey

@click.command()
@click.option('--dmd', \
    default=0, \
    multiple=True, \
    help='Chooses which DMD method to use: ' \
    + 'DMD options are dmd,' \
    + 'dmdc,sdmd,sdmdc,havok_dmd,optimized dmd,sdmdc (forecasting)' \
    + ' corresponding to 1-7, (0 for no dmd)')
@click.option('--numwindows', \
    default=1, \
    help='Number of windows for DMD')
@click.option('--directory', \
    default='/media/removable/SD Card/Two-Temperature-Post-Processing/', \
    help='Directory containing the .mat files')
@click.option('--postprocess', \
    default=False,type=bool, \
    help='Whether to plot the post-processed data,'+ \
    'like Itor and T_avg')
@click.option('--filenames', \
    default=['exppsi_122385.mat'],multiple=True, \
    help='A list of all the filenames which '+ \
        'allows a large number of shots to be '+ \
        'compared')
@click.option('--freqs', \
    default=14.5,multiple=True, \
    help='A list of all the injector frequencies (kHz) which '+ \
        'correspond to the list of filenames')
@click.option('--imp', \
    default=0,type=int, \
    help='Number of IMP signals')
@click.option('--fir', \
    default=False,type=bool, \
    help='Flag to use the FIR signal or not')
@click.option('--ids', \
    default=False,type=bool, \
    help='Flag to use the IDS signals or not')
@click.option('--limits', \
    default=(22.7,28.5),type=(float,float), \
    help='Time limits of the discharge')
@click.option('--nprocs', \
    default=1,type=int, \
    help='Number of processors to use (only relevant for oDMD)')
@click.option('--trunc', \
    default=10,type=int, \
    help='Where to truncate the SVD')

## Main program that accepts python 'click' command line arguments.
## Note that options with multiple=true must have multiple values added
## separately so that the command line command would be --dmd 1 --dmd 2
## rather than --dmd 1 2 or something. This format could also be done
## by declaring --dmd to be of type (int,int). If a description
## of the various click options is desired, just type
## python HITSI --help
def analysis(dmd,numwindows,directory,postprocess,filenames,freqs, \
    imp, fir, ids, limits, nprocs, trunc):
    is_HITSI3 = False
    if(len(filenames[0])==9):
        is_HITSI3=True

    filenames=np.atleast_1d(filenames)
    freqs=np.atleast_1d(freqs)
    print(dmd,numwindows,directory, \
        postprocess,filenames,freqs,limits)

    total = []
    for i in range(len(filenames)):
        filename = filenames[i]
        f_1 = np.atleast_1d(freqs[i])
        if filenames[i][0:10]=='Psi-Tet-2T':
            temp_dict = loadshot('Psi-Tet-2T',directory, \
                int(f_1),True,True,is_HITSI3,limits)
        elif filenames[i][0:3]=='Psi':
            temp_dict = loadshot('Psi-Tet',directory, \
                int(f_1),True,False,is_HITSI3,limits)
        else:
            temp_dict = loadshot(filename,directory, \
                np.atleast_1d(int(f_1)),False,False, \
                is_HITSI3,limits)
        if imp == 0:
            temp_dict['use_IMP'] = False
        else:
            temp_dict['use_IMP'] = True
            temp_dict['num_IMPs'] = imp
        temp_dict['use_FIR'] = fir
        temp_dict['use_IDS'] = ids
        temp_dict['nprocs'] = nprocs
        temp_dict['trunc'] = trunc
        temp_dict['f_1'] = f_1
        total.append(temp_dict)

    total = np.asarray(total).flatten()
    for k in range(len(dmd)):
        if dmd[k] != 0:
            if k == 0:
                for i in range(len(filenames)):
                    SVD(total[i])
            if dmd[k] > 0 and dmd[k] < 4:
                DMD_slide(total,numwindows,dmd[k])
            elif dmd[k] == 4:
                DMD_forecast(total[0])
            else:
                print('Invalid --dmd option, will assume no dmd')
    if len(dmd) >= 1:
        make_reconstructions(total[0],dmd)
        toroidal_modes_imp(total[0],freqs[0],dmd[1])
        toroidal_modes_sp(total[0],freqs[0],dmd[1])
        poloidal_modes(total[0],freqs[0],dmd[1])
        toroidal_plot(total[0],dmd[1])
        if postprocess:
            dmd_animation(total[0],freqs[0],numwindows,dmd)

if __name__ == '__main__':
    analysis()
    #sortby = SortKey.CUMULATIVE
    #cProfile.run('analysis()',sort=sortby)
