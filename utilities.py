## @package utilities
## Defines various functions for smoothing, calculating
## fourier transforms, SVD, and so on.
from plot_attributes import *
from map_probes import \
    sp_name_dict, dead_probes, \
    imp_phis8, imp_phis32, midphi
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
# @param dict A psi-tet dictionary
def SVD(dict):
    t0 = dict['t0']
    tf = dict['tf']
    data = np.vstack((dict['curr01'],dict['curr02']))
    if dict['is_HITSI3'] == True:
        data = np.vstack((data,dict['curr03']))
    #data = np.vstack((data,dict['flux01']))
    #data = np.vstack((data,dict['flux02']))
    data = np.vstack((data,dict['sp_Bpol']))
    data = np.vstack((data,dict['sp_Btor']))
    getshape = np.shape(data)[0]
    if dict['use_IMP']:
        dict['imp_Bpol'] = np.nan_to_num(dict['imp_Bpol'])[::1,:]
        dict['imp_Btor'] = np.nan_to_num(dict['imp_Btor'])[::1,:]
        dict['imp_Brad'] = np.nan_to_num(dict['imp_Brad'])[::1,:]
        dict['imp_Bpol'] = dict['imp_Bpol'][::1]
        dict['imp_Btor'] = dict['imp_Btor'][::1]
        dict['imp_Brad'] = dict['imp_Brad'][::1]
        data = np.vstack((data,dict['imp_Bpol']))
        shape1 = np.shape(dict['imp_Bpol'])[0]
        shape2 = np.shape(dict['imp_Btor'])[0]
        shape3 = np.shape(dict['imp_Brad'])[0]
        imp_pol_indices = np.linspace(0,shape1,shape1, \
            dtype = 'int')
        data = np.vstack((data,dict['imp_Btor']))
        imp_tor_indices = np.linspace(shape1,shape2+shape1,shape2, \
            dtype = 'int')
        data = np.vstack((data,dict['imp_Brad']))
        imp_rad_indices = np.linspace(shape1+shape2, \
            shape3+shape2+shape1,shape3, \
            dtype = 'int')

    # correct injector currents
    if dict['is_HITSI3'] == True:
        data[0:3,:] = data[0:3,:]*mu0
    else:
        data[0:2,:] = data[0:2,:]*mu0
    data = data[:,t0:tf]
    data_sub = -data #subtract_linear_trend(dict,data)
    u,s,v = np.linalg.svd(data_sub)
    v = np.conj(np.transpose(v))
    dict['SVD_data'] = data_sub
    dict['SP_data'] = data
    dict['U'] = u
    dict['S'] = s
    dict['V'] = v

## Identifies and subtracts a linear trend from each
## of the time signals contained in
## the SVD data associated with the dictionary 'dict'. This is
## to help DMD algorithms, since DMD does not deal well with
## non-exponential growth.
# @param dict A dictionary with SVD data
# @param data The SVD data matrix
# @returns data_subtracted The SVD data matrix
#  with the linear trend subtracted off
def subtract_linear_trend(dict,data):
    state_size = np.shape(data)[0]
    tsize = np.shape(data)[1]
    t0 = dict['t0']
    tf = dict['tf']
    time = dict['sp_time'][t0:tf]
    dt = dict['sp_time'][1] - dict['sp_time'][0]
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
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_sp(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset+size_bpol-32: \
            offset+size_bpol:2,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset+size_bpol-32: \
            offset+size_bpol:2,:]

    tsize = len(t_vec)
    phi = midphi
    nmax = 7
    amps = fourier_calc(nmax,tsize,Bfield_anom,phi)
    plt.figure(50000,figsize=(figx, figy))
    for m in range(nmax+1):
        plt.plot(t_vec*1000, \
            amps[m,:],label='n = '+str(m), \
            linewidth=lw)
            #plt.yscale('log')
    plt.legend(fontsize=ls,loc='upper right',ncol=2)
    plt.axvline(x=23.34,color='k')
    plt.xlabel('Time (ms)', fontsize=fs)
    plt.ylabel(r'$\delta B$', fontsize=fs)
    plt.title('Surface Probes', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_amps_sp.png')
    dict['toroidal_amps'] = amps
    plt.figure(60000,figsize=(figx, figy))
    plt.title('Surface Probes', fontsize=fs)
    plt.bar(range(nmax+1),amps[:,0]*1e4,color='r',edgecolor='k')
    plt.xlabel('Toroidal Mode',fontsize=fs)
    plt.ylabel('B (G)',fontsize=fs)
    ax = plt.gca()
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_sp_histogram.png')

## Computes the toroidal mode spectrum using
## a set of 8 or 32 IMPs
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def toroidal_modes_imp(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf-1]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    size_imp_bpol = np.shape(dict['imp_Bpol'])[0]
    size_imp_btor = np.shape(dict['imp_Btor'])[0]
    size_imp_brad = np.shape(dict['imp_Brad'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset+size_bpol+size_btor: \
            offset+size_bpol+size_btor+size_imp_bpol,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset+size_bpol+size_btor: \
            offset+size_bpol+size_btor+size_imp_bpol,:]

    print('sihi smooth freq = ',f_1)
    tsize = len(t_vec)
    num_IMPs = dict['num_IMPs']
    phis = np.zeros(160*num_IMPs)
    if num_IMPs == 8:
        imp_phis = imp_phis8
        nmax = 3
    elif num_IMPs == 32:
        imp_phis = imp_phis32
        nmax = 10
    else:
        print('Invalid number for the number of IMPs')
        exit()
    for i in range(num_IMPs):
        phis[i*160:(i+1)*160] = np.ones(160)*imp_phis[i]
    # subsample as needed
    phis = phis[::1]
    phis = phis[:len(phis)]
    amps = np.zeros((nmax+1,160,tsize))
    plt.figure(figsize=(figx+2, figy+2))
    for k in range(160):
        amps[:,k,:] = fourier_calc(nmax,tsize,Bfield_anom[k::160,:],phis[k::160])
        amax = np.max(np.max(amps[:,k,:]))
        if k % 10 == 0: 
          plt.subplot(4,4,int(k/10)+1)
          for m in range(nmax+1):
              plt.plot(t_vec*1000, \
                  amps[m,k,:]/amax, \
                  label='n = '+str(m))
          plt.ylim(-1,1)
          ax = plt.gca()
          ax.tick_params(axis='both', which='major', labelsize=ts-6)
          ax.tick_params(axis='both', which='minor', labelsize=ts-6)
          ax.set_xticks([])
          ax.set_yticks([-1,0,1])
    plt.savefig(out_dir+'toroidal_amps_imp.png')

    plt.figure(170000,figsize=(figx, figy))
    avg_amps = np.mean(abs(amps),axis=1)
    for m in range(nmax+1):
        plt.plot(t_vec*1000, \
            avg_amps[m,:],label='n = '+str(m), \
            linewidth=lw)
    plt.xlabel('Time (ms)', fontsize=fs)
    plt.title('Average of IMPs', fontsize=fs)
    plt.ylabel(r'$\delta B$', fontsize=fs)
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_imp.png')
    dict['toroidal_amps'] = avg_amps
    plt.figure(180000,figsize=(figx, figy))
    plt.title('Average of IMP Probes', fontsize=fs)
    plt.bar(range(nmax+1),avg_amps[:,0]*1e4,color='r',edgecolor='k')
    plt.xlabel('Toroidal Mode',fontsize=fs)
    plt.ylabel('B (G)',fontsize=fs)
    ax = plt.gca()
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.savefig(out_dir+'toroidal_avgamps_imp_histogram.png')

## Computes the poloidal mode spectrum for each
## of the four poloidal slices of the surface probes
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate which dmd method is used
def poloidal_modes(dict,dmd_flag):
    f_1 = dict['f_1']
    t0 = dict['t0']
    tf = dict['tf']
    t_vec = dict['sp_time'][t0:tf]
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    offset = 2
    if dict['is_HITSI3'] == True:
        offset = 3
    if dmd_flag == 2:
        Bfield_anom = dict['sparse_Bfield_anom'] \
            [offset:offset+size_bpol,:]
    elif dmd_flag == 3:
        Bfield_anom = dict['optimized_Bfield_anom'] \
            [offset:offset+size_bpol,:]
    tsize = len(t_vec)
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
            temp_B[k2, :] = Bfield_anom[j, :]
        if key[5:9] == 'P225' and \
            key[2:5] != 'L05' and key[2:5] != 'L06':
            temp_theta[k1] = sp_name_dict[key][3]
            k1 = k1 + 1
        if key[5] == 'P':
            j = j + 1
            k2 = k2 + 1
    phi_str = [r'$0^o$',r'$45^o$', \
        r'$180^0$',r'$225^o$']
    nmax = 7
    for i in range(4):
        plt.figure(80000,figsize=(figx, figy))
        B = temp_B[i::4,:]
        inds = ~np.all(B == 0, axis=1)
        B = B[inds]
        theta = temp_theta[np.where(inds)]
        amps = fourier_calc(nmax,tsize,B,theta)
        # can normalize by Bwall here
        # b1 = np.sqrt(int.sp.B_L04T000**2 + int.sp.B_L04P000**2)
        # b2 = np.sqrt(int.sp.B_L04T045**2 + int.sp.B_L04P045**2)
        # b3 = np.sqrt(int.sp.B_L04T180**2 + int.sp.B_L04P180**2)
        # b4 = np.sqrt(int.sp.B_L04T225**2 + int.sp.B_L04P225**2)
        # b0 = sihi_smooth((b1+b2+b3+b4)/4.0,t_vec,f_1)
        plt.subplot(2,2,i+1)
        for m in range(nmax+1):
            plt.plot(t_vec*1000, \
            amps[m,:],label='m = '+str(m),
            linewidth=lw)
        plt.title(r'$\phi$ = '+phi_str[i],fontsize=fs)
        if i == 0 or i == 2:
            plt.ylabel(r'$\delta B$', fontsize=fs)
        if i >= 2:
            plt.xlabel('Time (ms)',fontsize=fs)
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.savefig(out_dir+'poloidal_amps.png')
        dict['poloidal_amps'] = amps
        plt.figure(70000,figsize=(figx, figy))
        plt.subplot(2,2,i+1)
        plt.title(r'$\phi$ = '+phi_str[i],fontsize=fs)
        plt.bar(range(nmax+1),amps[:,0]*1e4,color='r',edgecolor='k')
        if i == 0 or i == 2:
            plt.ylabel('B (G)', fontsize=fs)
        if i >= 2:
            plt.xlabel('Poloidal Mode',fontsize=fs)
        ax = plt.gca()
        ax.set_xticks([0,1,2,3,4,5,6,7])
        ax.set_xticklabels(['0','1','2','3','4','5','6','7'])
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.savefig(out_dir+'poloidal_avgamps_sp_histogram.png')

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
