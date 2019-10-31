## @package dmd_plotting
## Defines functions for making plots
## and movies based on results from the
## DMD methods
from plot_attributes import *
from scipy.interpolate import griddata
from map_probes import \
    imp_rads, imp_phis8, imp_phis32

## Plots the power spectrum for the DMD
# @param b The bs determined from any DMD algorithm
# @param omega The complex DMD frequencies
# @param f_1 Injector frequency
# @param filename Name of the file corresponding to the shot
# @param typename type string indicating which algorithm is being used
def power_spectrum(b,omega,f_1,filename,typename):
    plt.figure(1000,figsize=(figx, figy+12))
    plt.subplot(4,1,4)
    f_k = np.imag(omega)/(pi*2*1000.0)
    delta_k = abs(np.real(omega)/(pi*2*1000.0))
    sort = np.argsort(f_k)
    power = (b[sort]*np.conj(b[sort])).astype('float')
    power = power/np.max(power)
    if typename=='DMD':
        plt.scatter(np.sort(f_k), \
            power,s=300,c='b',linewidths=3,edgecolors='k')
        #plt.plot(np.sort(f_k), \
        #    power,color='b',linewidth=lw,label=typename)
        plt.semilogy(np.sort(f_k), \
            power,color='b',linewidth=lw-4,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw,foreground='k'), \
            pe.Normal()])
    elif typename=='sparse DMD':
        plt.scatter(np.sort(f_k), \
            power,s=300,c='r',linewidths=3,edgecolors='k')
        #plt.plot(np.sort(f_k), \
        #    power,color='r',linewidth=lw,label=typename)
        plt.semilogy(np.sort(f_k), \
            power,color='r',linewidth=lw-4,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw,foreground='k'), \
            pe.Normal()])
    elif typename=='optimized DMD':
        plt.scatter(np.sort(f_k), \
            power,s=300,c='g',linewidths=3,edgecolors='k')
        #plt.plot(np.sort(f_k), \
        #    power,color='g',linewidth=lw,label=typename)
        plt.semilogy(np.sort(f_k), \
            power,color='g',linewidth=lw-4,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw,foreground='k'), \
            pe.Normal()])
    elif typename[9]=='=':
        plt.plot(np.sort(f_k), \
            power,'ro',markeredgecolor='k', \
            path_effects=[pe.Stroke(linewidth=lw,foreground='k'), \
            pe.Normal()])
        plt.semilogy(np.sort(f_k), \
            power,color='r',linewidth=lw-4, \
            label=typename, \
            path_effects=[pe.Stroke(linewidth=lw,foreground='k'), \
            pe.Normal()])
    plt.yscale('log')
    plt.ylabel(r'$|b_k|^2/|b_{max}|^2$',fontsize=fs)
    plt.xlabel(r'$f_k$ (kHz)',fontsize=fs)
    plt.xlim(-3*f_1,3*f_1)
    h=plt.ylabel(r'$|b_k|^2/|b_{max}|^2$',fontsize=fs)
    plt.xlabel(r'f (kHz)',fontsize=fs+4)
    plt.xlim(-3*f_1,3*f_1)
    plt.grid(True)
    ax = plt.gca()
    plt.ylim((1e-10,1e0))
    ax.set_xticks([-3*f_1,-2*f_1,-f_1,0, \
        f_1,2*f_1,3*f_1])
    ax.set_xticklabels([r'$-f_3^{inj}$',r'$-f_2^{inj}$',r'$-f_1^{inj}$',0, \
        r'$f_1^{inj}$',r'$f_2^{inj}$',r'$f_3^{inj}$'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.yticks([1e-10,1e-5,1e0])

    plt.figure(10000,figsize=(figx, figy))
    f_k = np.imag(omega)/(pi*2*1000.0)
    delta_k = abs(np.real(omega)/(pi*2*1000.0))
    sort = np.argsort(f_k)
    power = (b[sort]*np.conj(b[sort])).astype('float')
    power = power/np.max(power)
    if typename=='DMD':
        plt.semilogy(np.sort(f_k), \
            power,color='b',linewidth=lw,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()])
    elif typename=='sparse DMD':
        plt.semilogy(np.sort(f_k), \
            power,color='r',linewidth=lw,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()])
    elif typename=='optimized DMD':
        plt.semilogy(np.sort(f_k), \
            power,color='g',linewidth=lw,label=typename, \
            path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            pe.Normal()])
    elif typename[8]=='=':
        alphas = np.flip(np.linspace(0.1,1.0,4))
        if typename[14] == '-':
            alpha = alphas[0]
        if typename[14] == '0':
            alpha = alphas[1]
        if typename[14] == '1':
            alpha = alphas[2]
        if typename[14] == '2':
            alpha = alphas[3]
        plt.semilogy(np.sort(f_k), \
            power,color='r',linewidth=1, \
            label=typename,alpha=alpha, \
            path_effects=[pe.Stroke(linewidth=2,foreground='k'), \
            pe.Normal()])
        plt.scatter(np.sort(f_k), \
            power,s=100,c='k',edgecolors='k')
        plt.scatter(np.sort(f_k), \
            power,s=100,c='r',alpha=alpha,edgecolors='k')
    plt.yscale('log')
    plt.legend(edgecolor='k',facecolor='white',fontsize=ls,framealpha=1,loc='upper right')
    plt.xlim(-3*f_1,3*f_1)
    plt.xlim(-3*f_1,3*f_1)
    plt.grid(True)
    ax = plt.gca()
    plt.ylim((1e-19,1e1))
    ax.set_xticks([-3*f_1,-2*f_1,-f_1,0, \
        f_1,2*f_1,3*f_1])
    ax.set_xticklabels([r'$-f_3^{inj}$',r'$-f_2^{inj}$',r'$-f_1^{inj}$',0, \
        r'$f_1^{inj}$',r'$f_2^{inj}$',r'$f_3^{inj}$'])
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.grid(True)
    plt.yticks([1e-14,1e-7,1e0])
    plt.savefig(out_dir+filename)

## Plots real part vs imag part of the f_k frequencies
# @param b The bs determined from any DMD algorithm
# @param omega The complex DMD frequencies
# @param f_1 Injector frequency
# @param filename Name of the file corresponding to the shot
# @param typename type string indicating which algorithm is being used
def freq_phase_plot(b,omega,f_1,filename,typename):
    amp = abs(b)/np.max(abs(b))*1000 #/np.max(abs(b))
    camp = np.log(abs(b)/np.max(abs(b))) #/np.max(abs(b))
    for j in range(len(amp)):
        amp[j] = max(amp[j],100.0)
    sort = np.argsort(amp)
    amp = amp[sort]
    delta_k = np.real(omega[sort])/1000.0/(2*pi)
    f_k = np.imag(omega[sort])/1000.0/(2*pi)
    plt.figure(1000,figsize=(figx, figy+12))
    #plt.subplot(2,1,1)
    if typename=='DMD':
        plt.subplot(4,1,1)
        for snum in range(len(delta_k)):
            h0 = plt.scatter(f_k[snum],delta_k[snum],c='b',s=amp[snum], \
                linewidths=3,edgecolors='k', \
                label=typename,alpha=transparency)
    elif typename=='sparse DMD':
        plt.subplot(4,1,3)
        for snum in range(len(delta_k)):
            h0 = plt.scatter(f_k[snum],delta_k[snum],c='r',s=amp[snum], \
                linewidths=3,edgecolors='k', \
                label=typename,alpha=transparency)
        #plt.scatter(f_k,delta_k,c=amp,s=amp,cmap=plt.cm.get_cmap('Reds'), \
        #    linewidths=2,edgecolors='k', \
        #    label=typename,alpha=transparency)
        #plt.xlabel(r'f (kHz)',fontsize=fs)
    elif typename=='optimized DMD':
        plt.subplot(4,1,2)
        for snum in range(len(delta_k)):
            h0 = plt.scatter(f_k[snum],delta_k[snum],c='g',s=amp[snum], \
                linewidths=3,edgecolors='k', \
                label=typename,alpha=transparency)
        #plt.scatter(f_k,delta_k,c=amp,s=300.0,cmap=plt.cm.get_cmap('Greens'), \
        #    linewidths=2,edgecolors='k', \
        #    label=typename,alpha=transparency)
    plt.legend([h0.get_label()],edgecolor='k',facecolor='lightgrey',fontsize=ts,loc='lower right')
    plt.ylim(-1e3,1e0)
    plt.yscale('symlog',linthreshy=1e-2)
    ax = plt.gca()
    ax.set_yticks([-1e3,-1,-1e-2,1e-2,1e0])
    plt.axhline(y=0,color='k',linewidth=3,linestyle='--')
    plt.ylabel(r'$\delta_k$ (kHz)',fontsize=fs+4)
    plt.xlim(-120,120)
    #plt.xlim(-f_1*3,f_1*3)
    plt.grid(True)
    ax.set_xticks([-120,-5*f_1,-3*f_1,-f_1, \
        f_1,3*f_1,5*f_1,120])
    #ax.set_xticks([-3*f_1,-2*f_1,-f_1, \
    #    0,f_1,2*f_1,3*f_1])
    ax.set_xticklabels([])
    #ax.set_xticklabels(['-100',r'$-f_5$',r'$-f_3$',r'$-f_1$', \
    #    '0',r'$f_1$',r'$f_3$',r'$f_5$','100'])
    #ax.set_xticklabels([r'$-f_3$',r'$-f_2$',r'$-f_1$', \
    #    '0',r'$f_1$',r'$f_2$',r'$f_3$'])
    #ax.tick_params(axis='x', which='major', labelsize=ts)
    #ax.tick_params(axis='x', which='minor', labelsize=ts)
    ax.tick_params(axis='y', which='major', labelsize=ts)
    ax.tick_params(axis='y', which='minor', labelsize=ts)
    plt.savefig(out_dir+filename)

## Shows reconstructions using the DMD methods
## of a particular SP and a particular IMP probe
# @param dict A psi-tet dictionary
# @param dmd_flag Flag to indicate what type of dmd algorithm is being used
def make_reconstructions(dict,dmd_flag):
    t0 = dict['t0']
    tf = dict['tf']
    dictname = dict['filename']
    data = dict['SVD_data']
    size_bpol = np.shape(dict['sp_Bpol'])[0]
    size_btor = np.shape(dict['sp_Btor'])[0]
    index = size_bpol
    if dict['num_IMPs'] == 8:
    	imp_index = size_bpol+size_btor+1
    elif dict['num_IMPs'] == 32:
    	imp_index = size_bpol+size_btor+80
    inj_index = 2
    if dict['is_HITSI3']:
        inj_index = 3
    time = dict['sp_time'][t0:tf]*1000
    tsize = len(time)
    plt.figure(2000,figsize=(figx, figy))
    if dmd_flag==1:
        plt.subplot(3,1,1)
        plt.title(dict['filename'][7:13]+', Probe: B_L01T000', \
            fontsize=fs)
        reconstr = dict['Bfield']
        labelstring = 'DMD'
        color = 'b'
    elif dmd_flag==2:
        plt.subplot(3,1,2)
        reconstr = dict['sparse_Bfield']
        labelstring = 'sparse DMD'
        color = 'r'
    elif dmd_flag==3:
        plt.subplot(3,1,3)
        reconstr = dict['optimized_Bfield']
        labelstring = 'optimized DMD'
        color = 'g'
        plt.xlabel('Time (ms)',fontsize=fs)

    plt.plot(time, \
        data[index+inj_index,:]*1e4,'k',linewidth=lw)
    plt.plot(time[:tsize-1], \
        reconstr[index+inj_index,:tsize-1]*1e4,color,\
        label=labelstring+' reconstruction',linewidth=lw, \
        path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
        pe.Normal()])
    plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=ts)
    ax.tick_params(axis='both', which='minor', labelsize=ts)
    plt.legend(edgecolor='k',facecolor='lightgrey',fontsize=ls,loc='upper left')
    plt.ylabel('B (G)',fontsize=fs)
    #plt.ylim((-150,300))
    #ax.set_yticks([-150,0,150,300])
    plt.ylim((-500,600))
    ax.set_yticks([-500,0,500])
    plt.savefig(out_dir+'reconstructions'+str(dictname[:len(dictname)-4])+'_sp.png')

    plt.figure(3000,figsize=(figx, figy))
    if dmd_flag==1:
        plt.subplot(3,1,1)
        plt.title('BIG-HIT, Probe: IMP #8',fontsize=fs)
        reconstr = dict['Bfield']
        labelstring = 'DMD'
        color = 'b'
    elif dmd_flag==2:
        plt.subplot(3,1,2)
        reconstr = dict['sparse_Bfield']
        labelstring = 'sparse DMD'
        color = 'r'
    elif dmd_flag==3:
        plt.subplot(3,1,3)
        reconstr = dict['optimized_Bfield']
        labelstring = 'optimized DMD'
        color = 'g'
        plt.xlabel('Time (ms)',fontsize=fs)

    if dict['use_IMP']:
        plt.plot(time, \
            data[imp_index+inj_index,:]*1e4,'k',linewidth=3)
        plt.plot(time[:tsize-1], \
            reconstr[imp_index+inj_index,:tsize-1]*1e4,color,\
            label=labelstring+' reconstruction',linewidth=3) #, \
            #path_effects=[pe.Stroke(linewidth=lw+4,foreground='k'), \
            #pe.Normal()])
        plt.grid(True)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=ts)
        ax.tick_params(axis='both', which='minor', labelsize=ts)
        plt.legend(edgecolor='k',facecolor='lightgrey',fontsize=ls,loc='upper left')
        plt.ylabel('B (G)',fontsize=fs)
        plt.savefig(out_dir+'reconstructions'+str(dictname[:len(dictname)-4])+'_imp.png')

## Makes (R,phi) contour plots of B_theta (poloidal B field)
# @param dict A psi-tet dictionary
# @param dmd_flag which DMD method to use
def toroidal_plot(dict,dmd_flag):
    num_IMPs = dict['num_IMPs']
    t0 = dict['t0']
    tf = dict['tf']
    time = dict['sp_time'][t0:tf]*1000.0
    tsize = len(time)
    tstep = 500
    if dmd_flag == 3:
        tstep = 1
    FPS = 4
    offset = 2
    if dict['is_HITSI3']:
        offset = 3
    bpol_size = np.shape(dict['sp_Bpol'])[0]
    btor_size = np.shape(dict['sp_Btor'])[0]
    bpol_imp_size = np.shape(dict['imp_Bpol'])[0]
    phis_imp = np.zeros(160*num_IMPs)
    rads_imp = np.zeros(160*num_IMPs)
    for i in range(num_IMPs):
        if num_IMPs == 8:
          phis_imp[i*160:(i+1)*160] = np.ones(160)*imp_phis8[i]
          skip = 40
        elif num_IMPs == 32:
          phis_imp[i*160:(i+1)*160] = np.ones(160)*imp_phis32[i]
          skip = 1
        else:
          print('Invalid number of IMPs, exiting')
          exit()
        rads_imp[i*160:(i+1)*160] = np.ones(160)*imp_rads
    if dmd_flag == 1:
        bpol_f0_imp = dict['Bfield_f0'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f1_imp = dict['Bfield_f1'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f2_imp = dict['Bfield_f2'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f3_imp = dict['Bfield_f3'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_imp = dict['Bfield'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_kink_imp = dict['Bfield_kink'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
    elif dmd_flag == 2:
        bpol_f0_imp = dict['sparse_Bfield_f0'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f1_imp = dict['sparse_Bfield_f1'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f2_imp = dict['sparse_Bfield_f2'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f3_imp = dict['sparse_Bfield_f3'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_imp = dict['sparse_Bfield'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_kink_imp = dict['sparse_Bfield_kink'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
    elif dmd_flag == 3:
        bpol_f0_imp = dict['optimized_Bfield_f0'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f1_imp = dict['optimized_Bfield_f1'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f2_imp = dict['optimized_Bfield_f2'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_f3_imp = dict['optimized_Bfield_f3'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_imp = dict['optimized_Bfield'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]
        bpol_kink_imp = dict['optimized_Bfield_kink'] \
            [offset+bpol_size+btor_size: \
            offset+bpol_size+btor_size+bpol_imp_size,:]

    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    fig = plt.figure(figsize=(figx, figy))
    rimp = rads_imp[::skip]
    pimp = phis_imp[::skip]
    if num_IMPs==8:
        bindices = slice(0,29,4)
        indices = list(range(0,32))
        del indices[bindices]
        rimp = rimp[indices]
        pimp = pimp[indices]
    rorig = np.ravel([rimp, rimp, rimp])
    phiorig = np.ravel([pimp-2*pi, pimp, pimp+2*pi])
    midplanePhi = np.linspace(-2*pi,4*pi,len(imp_rads)*3)
    midplaneR, midplanePhi = np.meshgrid(imp_rads,midplanePhi)
    moviename = out_dir+'toroidal_Rphi_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour.eps')
    plt.savefig(out_dir+'contour.pdf')

    bpol_imp = bpol_imp - bpol_f1_imp - bpol_f0_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_subtracted_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_subtracted.eps')
    plt.savefig(out_dir+'contour_subtracted.pdf')

    bpol_imp = bpol_kink_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_kink_reconstruction.mp4'
    ani = animation.FuncAnimation( \
       fig, update_tor_Rphi, range(0,tsize,tstep), \
       fargs=(movie_bpol,midplaneR,midplanePhi, \
       rorig,phiorig,time),repeat=False, \
       interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tsize-2,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_kink.eps')
    plt.savefig(out_dir+'contour_kink.pdf')

    #bpol_imp = bpol_kink_imp
    #movie_bpol = np.vstack((bpol_imp,bpol_imp))
    #movie_bpol = np.vstack((movie_bpol,bpol_imp))
    #midplaneR, midplanePhi = np.meshgrid(imp_rads[60:120],midplanePhi)
    #moviename = out_dir+'toroidal_Rphi_kink_zoomed_reconstruction.mp4'
    #ani = animation.FuncAnimation( \
    #   fig, update_tor_Rphi, range(0,tsize,tstep), \
    #   fargs=(movie_bpol,midplaneR,midplanePhi, \
    #   rorig,phiorig,time),repeat=False, \
    #   interval=100, blit=False)
    #ani.save(moviename,fps=FPS)

    bpol_imp = bpol_f0_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_f0_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f0.eps')
    plt.savefig(out_dir+'contour_f0.pdf')

    bpol_imp = bpol_f1_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_f1_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f1.eps')
    plt.savefig(out_dir+'contour_f1.pdf')

    bpol_imp = bpol_f2_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_f2_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f2.eps')
    plt.savefig(out_dir+'contour_f2.pdf')

    bpol_imp = bpol_f3_imp
    movie_bpol = np.vstack((bpol_imp,bpol_imp))
    movie_bpol = np.vstack((movie_bpol,bpol_imp))
    moviename = out_dir+'toroidal_Rphi_f3_reconstruction.mp4'
    ani = animation.FuncAnimation( \
        fig, update_tor_Rphi, range(0,tsize,tstep), \
        fargs=(movie_bpol,midplaneR,midplanePhi, \
        rorig,phiorig,time),repeat=False, \
        interval=100, blit=False)
    ani.save(moviename,fps=FPS)
    update_tor_Rphi(tstep,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f3.eps')
    plt.savefig(out_dir+'contour_f3.pdf')
    update_tor_Rphi(tstep+5,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f3_1.eps')
    plt.savefig(out_dir+'contour_f3_1.pdf')
    update_tor_Rphi(tstep+10,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f3_2.eps')
    plt.savefig(out_dir+'contour_f3_2.pdf')
    update_tor_Rphi(tstep+15,movie_bpol,midplaneR,midplanePhi,rorig,phiorig,time)
    plt.savefig(out_dir+'contour_f3_3.eps')
    plt.savefig(out_dir+'contour_f3_3.pdf')

## Update function for FuncAnimation object
## for the (R,phi) contour plots
# @param frame A movie frame number
# @param Bpol Poloidal B in the plane
# @param midplaneR Radial coordinates where we interpolate
# @param midplanePhi Toroidal coordinates where we interpolate
# @param R Radial coordinates of the probes
# @param phi Toroidal coordinates of the probes
# @param time Array of times
def update_tor_Rphi(frame,Bpol,midplaneR,midplanePhi,R,phi,time):
    print(frame)
    plt.clf()
    #plt.xlabel('R (m)',fontsize=fs+10)
    #h = plt.ylabel(r'$\phi$',fontsize=fs+10)
    #h.set_rotation(0)
    #plt.title('Time = '+'{0:.3f}'.format(time[frame])+' ms',fontsize=fs)
    ax = plt.gca()
    # plot the probe locations
    #plt.plot(R, phi,'ko',markersize=5,label='Probes')
    #plt.plot([(1.0+0.625)/2.0,(1.0+0.625)/2.0], \
    #    [pi/8.0,pi+pi/8.0],'co',markersize=ms+8, \
    #    markeredgecolor='k',label='X Injector Mouths')
    #plt.plot([(1.0+0.625)/2.0,(1.0+0.625)/2.0], \
    #    [pi/2.0+pi/8.0,3*pi/2.0+pi/8.0],'yo', \
    #    markersize=ms+8,markeredgecolor='k',label='Y Injector Mouths')
    #ax.set_yticks([])
    ax.set_yticks([0,pi/2,pi,3*pi/2,2*pi])
    ax.set_yticklabels(['','',r'$\pi$','',r'$2\pi$'])
    ax.tick_params(axis='x', which='major', labelsize=fs+30)
    ax.tick_params(axis='x', which='minor', labelsize=fs+30)
    ax.tick_params(axis='y', which='major', labelsize=fs+30)
    ax.tick_params(axis='y', which='minor', labelsize=fs+30)
    Bpol_frame = Bpol[:,frame]
    grid_bpol = np.asarray( \
        griddata((R,phi),Bpol_frame,(midplaneR,midplanePhi),'cubic'))
    v = np.logspace(-3,0,10)
    v = np.ravel([-np.flip(v),v])
    grid_bpol = grid_bpol/np.max(np.max(abs(np.nan_to_num(grid_bpol))))
    contour = plt.contourf(midplaneR,midplanePhi, \
        grid_bpol,v,cmap=colormap,label=r'$B_\theta$', \
        norm=colors.SymLogNorm(linthresh=1e-3,linscale=1e-3))
    cbar = plt.colorbar(ticks=v,extend='both')
    cbar.ax.tick_params(labelsize=ts)
    cbar.ax.set_yticks([-1, -0.1, -0.01, -0.001, \
        0.001,0.01,0.1,1])
    cbar.ax.set_yticklabels(['-1','','', '-0.1','','', '-0.01', '', \
        '','','','','','0.01','','','0.1','','','1'])
    #ax.set_xticks([])
    ax.set_xticks([0,0.25,0.5,0.75,1.0,1.25])
    ax.set_xticklabels(['0','','0.5','','1',''])
    ax.fill_between([1.052,1.2849],0,2*pi,facecolor='lightgrey')
    ax.fill_between([0.0,0.368],0,2*pi,facecolor='lightgrey')
    #ax.set_xticks([0.37,0.7,1.05])
    #ax.set_xticklabels([0.37,0.7,1.05])
    #plt.legend(edgecolor='k',facecolor='white',fontsize=50,loc='lower left',
    #    framealpha=1.0)
    plt.ylim((0,2*pi))
    plt.xlim(0,1.2849)
