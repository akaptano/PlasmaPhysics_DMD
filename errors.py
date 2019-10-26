from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

plt.figure(figsize=(14,10))
plt.grid(True)
x = [1,3,5,10,20,40,60,80,100,120,140,160]
odmdy = [0.1952,0.1015,0.07,0.0424,0.03385, \
    0.02687,0.01973,0.01481,0.01236,0.01258,0.009,0.142]
#odmdf = [0.1988,0.16028,0.1437,0.1380,0.14247, \
#    0.1442,0.144,0.1443,0.1443,0.1445,0.1446]
#dmdf = [0.1971,0.1868,0.1755,0.1593,0.1559, \
#    0.161,0.1623,0.1637,0.1637,0.1639,0.1638]
dmdy = [0.1956,0.11417,0.1036,0.070111,0.06403, \
    0.05681,0.05298,0.05108,0.04732,0.04604,0.04488,0.0448]
#plt.xlabel('truncation number r',fontsize=40)
#h = plt.ylabel(r'$\epsilon$',fontsize=70)
#h.set_rotation(0)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.1,0.5)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)
l1 = plt.plot(x,dmdy,'b',linewidth=6,label='DMD reconstruction')
#l4 = plt.plot(x[:len(x)-1],dmdf,'b--',linewidth=6,label='DMD forecast')
l2 = plt.plot(x,odmdy,'g',linewidth=6,label='optimized DMD reconstruction')
#l5 = plt.plot(x[:len(x)-1],odmdf,'g--',linewidth=6,label='optimized DMD forecast')
l7 = plt.plot(x,0.044*np.ones(len(x)),color='b',linewidth=6,linestyle='--',label=r'DMD reconstruction error, r = 140')
plt.yscale('log')
plt.xscale('log')
#ax.set_xticklabels(x)
#sdmdf = [0.163,0.1627,0.1538,0.14036,0.1007,0.2726]
sdmdy = [0.051084,0.06588,0.1151,0.2773] 
gamma = [0.1,1.0,10.0,100.0]
ax = plt.gca()
ax.annotate('sparse DMD converges\nto traditional DMD', xy=(1.4, 6e-2),fontsize=26, xytext=(1.6, 0.3), \
    arrowprops=dict(facecolor='black', shrink=0.05))
#ax.annotate(r'only $f_0$ mode is nonzero', xy=(92, 3e-1),fontsize=20, xytext=(10, 4.5e-1), \
#    arrowprops=dict(facecolor='black', shrink=0.05))
ax2 = ax.twiny()
plt.grid(True)
ax2.set_xlim(ax.get_xlim())
#ax2.set_xlabel(r'sparsity promotion $\gamma$',fontsize=40)
ax2.set_xticks([1e-1,1e0,1e1,1e2])
plt.xscale('log')
plt.ylim(1e-3,1)
ax2.tick_params(axis='both', which='major', labelsize=30)
ax2.tick_params(axis='both', which='minor', labelsize=30)
ax2.yaxis.set_label_coords(-0.1,1.02)
ax2.tick_params(axis="x",which='both',direction="in", pad=-40)
ax2.xaxis.set_label_coords(0.5,1.03)
plt.xlim(5e-2,1.5e2)
ax2.set_xticks([1e-1,1e0,1e1,1e2])
ax2.spines['top'].set_color('red')
ax2.xaxis.label.set_color('red')
ax2.tick_params(axis='x', colors='red')
l3 = plt.plot(gamma,sdmdy,'r',linewidth=6,label='sparse DMD reconstruction, r = 140')
alphas = np.flip(np.linspace(0.1,1.0,len(gamma)))
for i in range(len(gamma)):
  plt.plot(gamma[i],sdmdy[i],'k*',markersize=34)
  plt.plot(gamma[i],sdmdy[i],'r*',markersize=34,markeredgecolor='k',alpha=alphas[i])
#l6 = plt.plot(gamma,sdmdf,'r--',linewidth=6,label='sparse DMD forecast, r = 140')
l = l1+l2+l3+l7
#l = l1+l4+l2+l5+l3+l6+l7
labs = [lines.get_label() for lines in l]
plt.legend(l,labs,edgecolor='k',facecolor='white',fontsize=26,loc='lower left')
ax2.annotate('Poor guess', xy=(1e2, 1.5e-1),fontsize=26, xytext=(1e1, 4e-1), \
    arrowprops=dict(facecolor='black', shrink=0.05))
#ax.set_xticklabels(x)
plt.savefig('Pictures/error.png')
plt.savefig('Pictures/error.eps')
plt.savefig('Pictures/error.pdf')
plt.savefig('Pictures/error.svg')
plt.show()
