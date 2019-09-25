## @package utilities
## Defines various functions for smoothing, calculating
## fourier transforms, SVD, and so on.
from plot_attributes import *
from map_probes import \
    sp_name_dict, dead_probes, \
    imp_phis8, imp_phis32, midphi, \
    imp_rads
from scipy.stats import linregress

## Python equivalent of the sihi_smooth function found in older
## matlab scripts. This does a boxcar average.
## Code simplified to only work for real-valued signals.
# @param y Signal in time
# @param time Time base associated with y
# @param f_1 Injector Frequency with which to apply the smoothing
# @returns x Smoothed signal in time
def sihi_smooth(y, time, f_1):
    injCyc = 1.0 / (1000.0 * f_1)
    Navg = 100
    Navg2 = int(Navg / 2.0)
    # make it 100 time points per injector cycle
    tint = np.linspace(time[0], time[len(time) - 1],
        int((time[len(time) - 1]-time[0]) / (injCyc / Navg)))
    yint = np.interp(tint, time, y)
    xint = np.zeros(len(tint))
    xint[0:Navg2] = np.mean(yint[0:Navg])

    for it in range(Navg2,len(tint) - Navg2):
        xint[it] = xint[it - 1] + \
            (yint[it + Navg2] - yint[it - Navg2]) / Navg

    xint[0:Navg2] = xint[Navg2]
    xint[len(tint) - Navg2:len(tint)] = \
        xint[len(tint) - Navg2 - 1]
    x = np.interp(time, tint, xint)
    x[len(x) - 1] = x[len(x) - 2]
    x[np.asarray(np.isnan(x)).nonzero()] = 0
    return x

## Performs a SVD of the data in a psi-tet dictionary.
## Has dmd_flags to control which data is put into the matrix
## for the SVD.
# @param psi_dict A psi-tet dictionary
def SVD(psi_dict):
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    data = np.vstack((psi_dict['curr01'],psi_dict['curr02']))
    if psi_dict['is_HITSI3'] == True:
        data = np.vstack((data,psi_dict['curr03']))
    #data = np.vstack((data,psi_dict['flux01']))
    #data = np.vstack((data,psi_dict['flux02']))
    data = np.vstack((data,psi_dict['sp_Bpol']))
    data = np.vstack((data,psi_dict['sp_Btor']))
    getshape = np.shape(data)[0]
    if psi_dict['use_IMP']:
        if psi_dict['num_IMPs'] == 8:
            skip = 40
            psi_dict['imp_Bpol'] = np.nan_to_num(psi_dict['imp_Bpol'])[::skip,:]
            psi_dict['imp_Btor'] = np.nan_to_num(psi_dict['imp_Btor'])[::skip,:]
            psi_dict['imp_Brad'] = np.nan_to_num(psi_dict['imp_Brad'])[::skip,:]
            bindices = slice(0,29,4)
            indices = list(range(0,32))
            del indices[bindices]
            psi_dict['imp_Bpol'] = psi_dict['imp_Bpol'][indices,:]
            psi_dict['imp_Btor'] = psi_dict['imp_Btor'][indices,:]
            psi_dict['imp_Brad'] = psi_dict['imp_Brad'][indices,:]
        if psi_dict['num_IMPs'] == 32:
            skip = 1
            psi_dict['imp_Bpol'] = np.nan_to_num(psi_dict['imp_Bpol'])[::skip,:]
            psi_dict['imp_Btor'] = np.nan_to_num(psi_dict['imp_Btor'])[::skip,:]
            psi_dict['imp_Brad'] = np.nan_to_num(psi_dict['imp_Brad'])[::skip,:]
        data = np.vstack((data,psi_dict['imp_Bpol']))
        shape1 = np.shape(psi_dict['imp_Bpol'])[0]
        shape2 = np.shape(psi_dict['imp_Btor'])[0]
        shape3 = np.shape(psi_dict['imp_Brad'])[0]
        imp_pol_indices = np.linspace(0,shape1,shape1, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Btor']))
        imp_tor_indices = np.linspace(shape1,shape2+shape1,shape2, \
            dtype = 'int')
        data = np.vstack((data,psi_dict['imp_Brad']))
        imp_rad_indices = np.linspace(shape1+shape2, \
            shape3+shape2+shape1,shape3, \
            dtype = 'int')

    # correct injector currents
    if psi_dict['is_HITSI3'] == True:
        data[0:3,:] = data[0:3,:]*mu0
    else:
        data[0:2,:] = data[0:2,:]*mu0
    # For forecasting
    #psi_dict['full_data'] = data[:,t0:tf]
    #data = data[:,t0:t0+int(tf/2)-1]
    data = data[:,t0:tf]
    noise = np.random.normal(0,5e-4,(np.shape(data)[0],np.shape(data)[1]))
    data_sub = -data #+noise
    #data_sub = subtract_linear_trend(psi_dict,data)
    u,s,v = np.linalg.svd(data_sub)
    v = np.conj(np.transpose(v))
    psi_dict['SVD_data'] = data_sub
    psi_dict['SP_data'] = data
    psi_dict['U'] = u
    psi_dict['S'] = s
    psi_dict['V'] = v

## Identifies and subtracts a linear trend from each
## of the time signals contained in
## the SVD data associated with the dictionary 'psi_dict'. This is
## to help DMD algorithms, since DMD does not deal well with
## non-exponential growth.
# @param psi_dict A dictionary with SVD data
# @param data The SVD data matrix
# @returns data_subtracted The SVD data matrix
#  with the linear trend subtracted off
def subtract_linear_trend(psi_dict,data):
    state_size = np.shape(data)[0]
    tsize = np.shape(data)[1]
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]
    dt = psi_dict['sp_time'][1] - psi_dict['sp_time'][0]
    data_subtracted = np.zeros((state_size,tsize))
    for i in range(state_size):
        slope, intercept, r_value, p_value, std_err = linregress(time,data[i,:])
        data_subtracted[i,:] = data[i,:] - (slope*time+intercept)
        if i == 10:
            plt.figure()
            plt.plot(time,data_subtracted[i,:],'g')
            plt.plot(time,slope*time+intercept,'b')
            plt.plot(time,data[i,:],'r')
            plt.savefig(out_dir+'linear_trend_test.png')
    return data_subtracted

## Computes the toroidal mode spectrum using the
## surface midplane gap probes
# @param psi_dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_sp(psi_dict,dmd_flag):
    f_1 = psi_dict['f_1']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    t_vec = psi_dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(psi_dict['sp_Bpol'])[0]
    size_btor = np.shape(psi_dict['sp_Btor'])[0]
    offset = 2
    tsize = len(t_vec)
    print(t_vec[0])
    phi = midphi
    nmax = 7
    if psi_dict['is_HITSI3'] == True:
        offset = 3
    for i in range(5):
        if dmd_flag == 1:
            ext = ''
            color = 'b'
        elif dmd_flag == 2:
            ext = 'sparse_'
            color = 'r'
        elif dmd_flag == 3:
            ext = 'optimized_'
            color = 'g'
        if i == 0:
            Bfield = psi_dict[ext+'Bfield_f0'] \
                [offset+size_bpol-32: \
                offset+size_bpol:2,:]
            labelhist = r'$f_0$'
        elif i == 1:
            Bfield = psi_dict[ext+'Bfield_f1'] \
                [offset+size_bpol-32: \
                offset+size_bpol:2,:]
            labelhist = r'$f_1^\mathrm{inj}$'
        elif i == 2:
            Bfield = psi_dict[ext+'Bfield_f2'] \
                [offset+size_bpol-32: \
                offset+size_bpol:2,:]
            labelhist = r'$f_2^\mathrm{inj}$'
        elif i == 3:
            Bfield = psi_dict[ext+'Bfield_f3'] \
                [offset+size_bpol-32: \
                offset+size_bpol:2,:]
            labelhist = r'$f_3^\mathrm{inj}$'
        elif i == 4:
            Bfield = psi_dict[ext+'Bfield_kink'] \
                [offset+size_bpol-32: \
                offset+size_bpol:2,:]
            labelhist = r'$f_{kink}$'
        amps = fourier_calc(nmax,tsize,Bfield,phi)
        if i <= 3:
            plt.figure(50000,figsize=(figx, figy))
            plt.subplot(4,1,i+1)
            if i+1==4:
                plt.xlabel('Time (ms)', fontsize=fs)
        else:
            plt.figure(50000+i,figsize=(figx, figy))
        for m in range(nmax+1):
            plt.plot(t_vec*1000, \
                amps[m,:],label='n = '+str(m), \
                linewidth=lw)
                #plt.yscale('log')
        plt.legend(edgecolor='k',facecolor='wheat',fontsize=ls,loc='upper right')
        plt.axvline(x=23.34,color='k')
        plt.ylabel(r'$\delta B$', fontsize=fs)
        #plt.title('Surface Probes', fontsize=fs)
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        if i <= 3:
            plt.savefig(out_dir+'toroidal_amps_sp.png')
            plt.savefig(out_dir+'toroidal_amps_sp.eps')
            plt.savefig(out_dir+'toroidal_amps_sp.pdf')
            plt.savefig(out_dir+'toroidal_amps_sp.svg')
        else:
            plt.savefig(out_dir+'toroidal_amps_sp_kink.png')
            plt.savefig(out_dir+'toroidal_amps_sp_kink.eps')
            plt.savefig(out_dir+'toroidal_amps_sp_kink.pdf')
            plt.savefig(out_dir+'toroidal_amps_sp_kink.svg')
        psi_dict['toroidal_amps'] = amps
        if i <= 3:
            plt.figure(60000,figsize=(figx, figy))
            plt.subplot(4,1,i+1)
            #for m in range(nmax+1):
            plt.bar(range(nmax+1),amps[:,0]*1e4,color='r',edgecolor='k', \
                label=labelhist)
            #plt.bar(range(8,11),0.0,color='r',edgecolor='k')
            ax = plt.gca()
            if i+1==4:
                plt.xlabel(r'$n_\phi$',fontsize=fs)
                ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ''])
            else:
                ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                ax.set_xticklabels([])
            ax.set_yticks([min(amps[:,0])*1e4,max(amps[:,0])*1e4])
            ax.set_yticklabels([int(min(amps[:,0])*1e4),int(max(amps[:,0])*1e4)])
        else:
            plt.figure(60000+i,figsize=(figx, figy))
            for m in range(nmax+1):
                plt.bar(m,amps[m,tsize-2]*1e4,edgecolor='k')
        #plt.title('Surface Probes', fontsize=fs)
        props = dict(boxstyle='round', facecolor='wheat', edgecolor='k', alpha=0.5)
        ax.text(0.92, 0.87, labelhist, transform=ax.transAxes, fontsize=40,
            verticalalignment='top', bbox=props)
        #plt.legend(edgecolor='k',facecolor='wheat',fontsize=ls,loc='upper right')
        h0 = plt.ylabel('B (G)',fontsize=ts)
        h0.set_rotation(0)
        ax = plt.gca()
        ax.yaxis.set_label_coords(-0.12,0.4)
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        if i <= 3:
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.png')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.eps')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.svg')
        else:
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram_kink.png')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram_kink.eps')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram_kink.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_sp_histogram_kink.svg')

## Computes the toroidal mode spectrum using
## a set of 8 or 32 IMPs
# @param psi_dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_imp(psi_dict,dmd_flag):
    f_1 = psi_dict['f_1']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    t_vec = psi_dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(psi_dict['sp_Bpol'])[0]
    size_btor = np.shape(psi_dict['sp_Btor'])[0]
    size_imp_bpol = np.shape(psi_dict['imp_Bpol'])[0]
    size_imp_btor = np.shape(psi_dict['imp_Btor'])[0]
    size_imp_brad = np.shape(psi_dict['imp_Brad'])[0]
    offset = 2
    if psi_dict['is_HITSI3'] == True:
        offset = 3
    tsize = len(t_vec)
    num_IMPs = psi_dict['num_IMPs']
    phis = np.zeros(160*num_IMPs)
    if num_IMPs == 8:
        imp_phis = imp_phis8
        nmax = 3
        skip = 40
    elif num_IMPs == 32:
        imp_phis = imp_phis32
        nmax = 10
        skip = 1
    else:
        print('Invalid number for the number of IMPs')
        exit()
    for i in range(num_IMPs):
        phis[i*160:(i+1)*160] = np.ones(160)*imp_phis[i]
    # subsample as needed
    phis = phis[::skip]
    orig_phis = phis[:len(phis)]
    for i in range(5):
        amps = np.zeros((nmax+1,160,tsize))
        subcount = 1
        if dmd_flag == 1:
            ext = ''
            color = 'b'
        elif dmd_flag == 2:
            ext = 'sparse_'
            color = 'r'
        elif dmd_flag == 3:
            ext = 'optimized_'
            color = 'g'
        if i == 0:
            Bfield = psi_dict[ext+'Bfield_f0'] \
                [offset+size_bpol+size_btor: \
                offset+size_bpol+size_btor+size_imp_bpol,:]
            labelhist = r'$f_0$'
        elif i == 1:
            Bfield = psi_dict[ext+'Bfield_f1'] \
                [offset+size_bpol+size_btor: \
                offset+size_bpol+size_btor+size_imp_bpol,:]
            labelhist = r'$f_1^\mathrm{inj}$'
        elif i == 2:
            Bfield = psi_dict[ext+'Bfield_f2'] \
                [offset+size_bpol+size_btor: \
                offset+size_bpol+size_btor+size_imp_bpol,:]
            labelhist = r'$f_2^\mathrm{inj}$'
        elif i == 3:
            Bfield = psi_dict[ext+'Bfield_f3'] \
                [offset+size_bpol+size_btor: \
                offset+size_bpol+size_btor+size_imp_bpol,:]
            labelhist = r'$f_3^\mathrm{inj}$'
        elif i == 4:
            Bfield = psi_dict[ext+'Bfield_kink'] \
                [offset+size_bpol+size_btor: \
                offset+size_bpol+size_btor+size_imp_bpol,:]
            labelhist = r'$f_{kink}$'

        if num_IMPs == 8:
            bindices = slice(0,29,4)
            indices = list(range(0,32))
            del indices[bindices]
            phis = orig_phis[indices]
            for k in range(3):
                amps[:,k,:] = fourier_calc(nmax,tsize,Bfield[k::3,:],phis[k::3])
                if i == 4:
                    plt.figure(figsize=(figx, figy))
                    amax = np.max(np.max(amps[:,k,:]))
                    for m in range(nmax+1):
                        plt.plot(t_vec*1000, \
                            amps[m,k,:]*1e4, \
                            label=r'$n_\phi$ = '+str(m), \
                            linewidth=3)
                    ax = plt.gca()
                    ax.tick_params(axis='both', which='major', labelsize=ts)
                    ax.tick_params(axis='both', which='minor', labelsize=ts)
                    plt.title('R = {0:.2f} m'.format(imp_rads[40*(k+1)]),fontsize=fs)
                    plt.xlabel('Time (ms)',fontsize=fs)
                    ax.set_xticks([26.8,27.1])
                    plt.ylim(-40,60)
                    plt.grid(True)
                    plt.ylabel('B (G)',fontsize=fs)
                    plt.savefig(out_dir+'toroidal_amps_imp'+str(k)+'.png')
                    plt.savefig(out_dir+'toroidal_amps_imp'+str(k)+'.eps')
                    plt.savefig(out_dir+'toroidal_amps_imp'+str(k)+'.pdf')
                    plt.savefig(out_dir+'toroidal_amps_imp'+str(k)+'.svg')
        elif num_IMPs == 32:
            for k in range(160):
                amps[:,k,:] = fourier_calc(nmax,tsize,Bfield[k::160,:],phis[k::160])
                if i == 4:
                    plt.figure(figsize=(figx, figy))
                    amax = np.max(np.max(amps[:,k,:]))
                    if k >= 74 and k < 90:
                        #plt.subplot(4,4,k+1)
                        plt.subplot(4,4,subcount)
                        for m in range(nmax+1):
                            plt.plot(t_vec*1000, \
                                amps[m,k,:]/amax, \
                                label='n = '+str(m), \
                                linewidth=3)
                        plt.ylim(-1,1)
                        ax = plt.gca()
                        ax.tick_params(axis='both', which='major', labelsize=ts-6)
                        ax.tick_params(axis='both', which='minor', labelsize=ts-6)
                        ax.set_yticks([])
                        if subcount >= 13:
                            plt.xlabel('Time (ms)',fontsize=ts)
                            ax.set_xticks([26.8,27.1])
                        else:
                            ax.set_xticks([])
                        subcount = subcount+1
            plt.savefig(out_dir+'toroidal_amps_imp_kink.png')
            plt.savefig(out_dir+'toroidal_amps_imp_kink.eps')
            plt.savefig(out_dir+'toroidal_amps_imp_kink.pdf')
            plt.savefig(out_dir+'toroidal_amps_imp_kink.svg')
        if i <= 3:
            plt.figure(170000,figsize=(figx, figy))
            plt.subplot(4,1,i+1)
        else:
            plt.figure(170000+i,figsize=(figx, figy))
        if num_IMPs == 8:
            avg_amps = np.mean(amps[:,0:4,:],axis=1)
        elif num_IMPs == 32:
            avg_amps = np.mean(amps,axis=1)
        #avg_amps = np.mean(abs(amps),axis=1)
        for m in range(nmax+1):
            plt.plot(t_vec*1000, \
                avg_amps[m,:]*1e4,label=r'$n_\phi =$ '+str(m), \
                linewidth=lw)
        plt.xlabel('Time (ms)', fontsize=fs)
        plt.title('Average of IMPs', fontsize=fs)
        h = plt.ylabel(r'$B_{kink}$ (G)', fontsize=fs)
        #plt.legend(fontsize=ls-10,loc='upper left')
        plt.grid(True)
        ax = plt.gca()
        ax.set_xticks([26.8,27.1])
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        if i <= 3:
            plt.savefig(out_dir+'toroidal_avgamps_imp.png')
            plt.savefig(out_dir+'toroidal_avgamps_imp.eps')
            plt.savefig(out_dir+'toroidal_avgamps_imp.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_imp.svg')
        else:
            plt.savefig(out_dir+'toroidal_avgamps_imp_kink.png')
            plt.savefig(out_dir+'toroidal_avgamps_imp_kink.eps')
            plt.savefig(out_dir+'toroidal_avgamps_imp_kink.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_imp_kink.svg')
        psi_dict['toroidal_amps'] = avg_amps
        if i <= 3:
            plt.figure(180000,figsize=(figx, figy))
            plt.subplot(4,1,i+1)
            #for m in range(11):
            #    if m < nmax+1:
            plt.bar(range(nmax+1),avg_amps[:,0]*1e4,color='r',edgecolor='k')
            #    else:
            #        plt.bar(m,0.0,color='r',edgecolor='k')
            ax = plt.gca()
            if i == 3:
                plt.xlabel(r'$n_\phi$',fontsize=fs)
                ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
                ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ''])
            else:
                ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                ax.set_xticklabels([])
            ax.set_yticks([min(avg_amps[:,0])*1e4,max(avg_amps[:,0])*1e4])
            ax.set_yticklabels([int(min(avg_amps[:,0])*1e4),int(max(avg_amps[:,0])*1e4)])
        else:
            plt.figure(180000+i,figsize=(figx, figy))
        #plt.title('Average of IMP Probes', fontsize=fs)
            for m in range(11):
                if m < nmax+1:
                    plt.bar(m,avg_amps[m,tsize-2]*1e4,edgecolor='k')
                else:
                    plt.bar(m,0.0,edgecolor='k')
            ax = plt.gca()
            ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10])

        props = dict(boxstyle='round', facecolor='wheat', edgecolor='k', alpha=0.5)
        ax.text(0.92, 0.87, labelhist, transform=ax.transAxes, fontsize=40,
            verticalalignment='top', bbox=props)
        h0 = plt.ylabel('B (G)',fontsize=ts)
        h0.set_rotation(0)
        ax = plt.gca()
        ax.yaxis.set_label_coords(-0.12,0.4)
        #if num_IMPs == 8:
        #    ax.set_xticks([0, 1, 2, 3])
        #if num_IMPs == 32:
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        if i <= 3:
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.png')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.eps')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.svg')
        else:
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram_kink.png')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram_kink.eps')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram_kink.pdf')
            plt.savefig(out_dir+'toroidal_avgamps_imp_histogram_kink.svg')

## Computes the poloidal mode spectrum for each
## of the four poloidal slices of the surface probes
# @param psi_dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def poloidal_modes(psi_dict,dmd_flag):
    f_1 = psi_dict['f_1']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    t_vec = psi_dict['sp_time'][t0:tf]
    size_bpol = np.shape(psi_dict['sp_Bpol'])[0]
    size_btor = np.shape(psi_dict['sp_Btor'])[0]
    offset = 2
    if psi_dict['is_HITSI3'] == True:
        offset = 3
    tsize = len(t_vec)
    nmax = 7
    for i in range(5):
        amps = np.zeros((nmax+1,4,tsize))
        subcount = 1
        if dmd_flag == 1:
            ext = ''
            color = 'b'
        elif dmd_flag == 2:
            ext = 'sparse_'
            color = 'r'
        elif dmd_flag == 3:
            ext = 'optimized_'
            color = 'g'
        if i == 0:
            Bfield = psi_dict[ext+'Bfield_f0'] \
                [offset:offset+size_bpol,:]
            labelhist = r'$f_0$'
        elif i == 1:
            Bfield = psi_dict[ext+'Bfield_f1'] \
                [offset:offset+size_bpol,:]
            labelhist = r'$f_1^\mathrm{inj}$'
        elif i == 2:
            Bfield = psi_dict[ext+'Bfield_f2'] \
                [offset:offset+size_bpol,:]
            labelhist = r'$f_2^\mathrm{inj}$'
        elif i == 3:
            Bfield = psi_dict[ext+'Bfield_f3'] \
                [offset:offset+size_bpol,:]
            labelhist = r'$f_3^\mathrm{inj}$'
        elif i == 4:
            Bfield = psi_dict[ext+'Bfield_kink'] \
                [offset:offset+size_bpol,:]
            labelhist = r'$f_{kink}$'
        # Find the poloidal gap probes
        k1 = 0
        k2 = 0
        j = 0
        B = np.zeros((16,tsize))
        theta = np.zeros(16)
        temp_B = np.zeros((64,tsize))
        temp_theta = np.zeros(16)
        for key in sp_name_dict.keys():
            if key in dead_probes:
                if key[5] == 'P':
                    k2 = k2 + 1
                continue
            if key[5] == 'P' and \
                key[2:5] != 'L05' and key[2:5] != 'L06':
                temp_B[k2, :] = Bfield[j, :]
            if key[5:9] == 'P225' and \
                key[2:5] != 'L05' and key[2:5] != 'L06':
                temp_theta[k1] = sp_name_dict[key][3]
                k1 = k1 + 1
            if key[5] == 'P':
                j = j + 1
                k2 = k2 + 1
        phi_str = [r'$0^o$',r'$45^o$', \
            r'$180^0$',r'$225^o$']
        for j in range(4):
            plt.figure(80000,figsize=(figx, figy))
            B = temp_B[j::4,:]
            inds = ~np.all(B == 0, axis=1)
            B = B[inds]
            theta = temp_theta[np.where(inds)]
            amps[:,j,:] = fourier_calc(nmax,tsize,B,theta)
        avg_amps = np.mean(amps,axis=1)
        # can normalize by Bwall here
        # b1 = np.sqrt(int.sp.B_L04T000**2 + int.sp.B_L04P000**2)
        # b2 = np.sqrt(int.sp.B_L04T045**2 + int.sp.B_L04P045**2)
        # b3 = np.sqrt(int.sp.B_L04T180**2 + int.sp.B_L04P180**2)
        # b4 = np.sqrt(int.sp.B_L04T225**2 + int.sp.B_L04P225**2)
        # b0 = sihi_smooth((b1+b2+b3+b4)/4.0,t_vec,f_1)
        if i <= 3:
            plt.subplot(4,1,i+1)
            for m in range(nmax+1):
                plt.plot(t_vec*1000, \
                amps[m,0,:]*1e4,label=r'$m_\theta$ = '+str(m),
                linewidth=lw)
            plt.title(r'$\phi$ = '+phi_str[i],fontsize=fs-10)
            if i == 0 or i == 2:
                plt.ylabel('B (G)', fontsize=fs-10)
            if i >= 2:
                plt.xlabel('Time (ms)',fontsize=fs-10)
            plt.grid(True)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=ts-10)
            ax.tick_params(axis='both', which='minor', labelsize=ts-10)
            plt.savefig(out_dir+'poloidal_amps.png')
            plt.savefig(out_dir+'poloidal_amps.eps')
            plt.savefig(out_dir+'poloidal_amps.pdf')
            plt.savefig(out_dir+'poloidal_amps.svg')
        psi_dict['poloidal_amps'] = amps
        if i <= 3:
            plt.figure(70000,figsize=(figx, figy))
            ax = plt.gca()
            plt.subplot(4,1,i+1)
            plt.bar(range(nmax+1),avg_amps[:,0]*1e4,color='r',edgecolor='k')
            ax = plt.gca()
            ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 9])
            if i==3:
                plt.xlabel(r'$m_\theta$',fontsize=fs)
                ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, ''])
            else:
                ax.set_xticklabels([])
            ax.set_yticks([min(avg_amps[:,0])*1e4,max(avg_amps[:,0])*1e4])
            ax.set_yticklabels([int(min(avg_amps[:,0])*1e4),int(max(avg_amps[:,0])*1e4)])
        else:
            plt.figure(70000+i,figsize=(figx, figy))
            plt.bar(range(nmax+1),avg_amps[:,tsize-2]*1e4,color='r',edgecolor='k')
        props = dict(boxstyle='round', facecolor='wheat', edgecolor='k', alpha=0.5)
        ax.text(0.92, 0.87, labelhist, transform=ax.transAxes, fontsize=40,
            verticalalignment='top', bbox=props)
        #plt.legend(edgecolor='k',facecolor='wheat',fontsize=ls,loc='upper right')
        h0 = plt.ylabel('B (G)',fontsize=ts)
        h0.set_rotation(0)
        ax = plt.gca()
        ax.yaxis.set_label_coords(-0.12,0.4)
        #ax.set_yticks([-50,0,100,200])
        #ax.set_xticks([0,1,2,3,4,5,6,7])
        #ax.set_xticklabels(['0','1','2','3','4','5','6','7'])
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        if i <= 3:
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.png')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.eps')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.pdf')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.svg')
        else:
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram_kink.png')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram_kink.eps')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram_kink.pdf')
            plt.savefig(out_dir+'poloidal_avgamps_sp_histogram_kink.svg')

## Performs the fourier calculation based on Wrobel 2011
# @param nmax The toroidal/poloidal number resolution
# @param tsize The number of time snapshots
# @param b Magnetic field signals of a toroidal/poloidal set of probes
# @param phi Toroidal/poloidal angles associated with the probes
# @returns amps The toroidal/poloidal mode amplitudes
def fourier_calc(nmax,tsize,b,phi):
    # Set up mode calculation- code adapted from JSW
    minvar = 1e-10 # minimum field variance for calcs.
    amps = np.zeros((nmax+1,tsize))
    phases = np.zeros((nmax+1,tsize))
    vardata = np.zeros(np.shape(b)) + minvar
        # Calculate highest nmax possible
    nprobes = np.shape(b)[0]
    mcoeff = np.zeros((2*nmax + 1,2*nmax + 1))
    for nn in range(nmax+1):
        for m in range(nmax+1):
            mcoeff[m, nn] = \
                sum(np.cos(m*phi) * np.cos(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax, nn] = \
                sum(np.sin(m*phi) * np.cos(nn*phi))

    for nn in range(1,nmax+1):
        for m in range(nmax+1):
            mcoeff[m,nn+nmax] = \
                sum(np.cos(m*phi) * np.sin(nn*phi))
        for m in range(1,nmax+1):
            mcoeff[m+nmax,nn+nmax] = \
                sum(np.sin(m*phi) * np.sin(nn*phi))

    asnbs    = np.zeros(2*nmax + 1)
    varasnbs = np.zeros(2*nmax + 1)
    rhs      = np.zeros(2*nmax + 1)
    veca     = np.zeros((nmax+1,tsize))
    vecb     = np.zeros((nmax,tsize))
    for m in range(tsize):
        bflds = b[:, m]
        varbflds = vardata[:, m]
        for nn in range(nmax+1):
            rhs[nn] = sum(bflds*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(bflds*np.sin(nn*phi))
        asnbs,g1,g2,g3 = np.linalg.lstsq(mcoeff,rhs)
        for nn in range(nmax+1):
            rhs[nn] = sum(np.sqrt(varbflds)*np.cos(nn*phi))
        for nn in range(1,nmax+1):
            rhs[nn + nmax] = sum(np.sqrt(varbflds)*np.sin(nn*phi))
        veca[0:nmax+1, m] = asnbs[0:nmax+1]
        vecb[0:nmax, m] = asnbs[nmax+1:2*nmax+1]
    amps[0,:] = veca[0, :]
    phases[0,:] = 0.0 * veca[0, :]

    for m in range(nmax):
        amps[m+1,:] = \
            np.sqrt(veca[m+1, :]**2 + vecb[m, :]**2)
        phases[m+1,:] = np.arctan2(vecb[m, :], veca[m+1, :])
    return amps

## Plots the toroidal current for a shot
# @param psi_dict A psi-tet dictionary
def plot_itor(psi_dict):
    itor = psi_dict['tcurr']/1000.0
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['time']*1000.0
    plt.figure(75000,figsize=(figx, figy))
    plt.plot(time,itor,color='orange',linewidth=lw)
    plt.legend(edgecolor='k',facecolor='wheat',fontsize=ls,loc='upper right')
    plt.axvline(x=time[t0],color='k')
    plt.axvline(x=time[tf],color='k')
    plt.xlabel('Time (ms)', fontsize=fs)
    plt.ylabel(r'$I_{tor}$ (kA)', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_current.png')
    plt.savefig(out_dir+'toroidal_current.eps')
    plt.savefig(out_dir+'toroidal_current.pdf')
    plt.savefig(out_dir+'toroidal_current.svg')

## Plots the BD chronos for a shot
# @param psi_dict A psi-tet dictionary
def plot_chronos(psi_dict):
    Vh = np.transpose(np.conj(psi_dict['V']))
    S = psi_dict['S']
    t0 = psi_dict['t0']
    tf = psi_dict['tf']
    time = psi_dict['sp_time'][t0:tf]*1000.0
    alphas = np.flip(np.linspace(0.3,1.0,3))
    plt.figure(85000,figsize=(figx, figy))
    for i in range(3):
        plt.plot(time,S[i]*Vh[i,:]*1e4/S[0],'m',linewidth=lw, alpha=alphas[i], \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()],label='Mode '+str(i+1))
    plt.legend(edgecolor='k',facecolor='wheat',fontsize=ls,loc='lower left')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    plt.xlabel('Time (ms)', fontsize=fs)
    h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}V_{ki}^*$', fontsize=fs)
    h.set_rotation(0)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD_chronos.png')
    plt.savefig(out_dir+'BD_chronos.eps')
    plt.savefig(out_dir+'BD_chronos.pdf')
    plt.savefig(out_dir+'BD_chronos.svg')
    plt.figure(95000,figsize=(figx, figy))
    plt.semilogy(range(1,len(S)+1),S/S[0],'mo',markersize=ms,markeredgecolor='k')
    plt.semilogy(range(1,len(S)+1),S/S[0],'m')
    #plt.axvline(x=time[t0],color='k')
    #plt.axvline(x=time[tf],color='k')
    plt.xlabel('Mode Number k', fontsize=fs)
    h = plt.ylabel(r'$\frac{\Sigma_{kk}}{\Sigma_{00}}$', fontsize=fs)
    h.set_rotation(0)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'BD.png')
    plt.savefig(out_dir+'BD.eps')
    plt.savefig(out_dir+'BD.pdf')
    plt.savefig(out_dir+'BD.svg')
