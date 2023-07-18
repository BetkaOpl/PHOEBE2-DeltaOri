#!/usr/bin/python3
# MCMC - corner plot

# Imported libraries
import corner
import numpy as np
import matplotlib.pyplot as plt

# Font setting
params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'figure.titleweight' : 'bold',
          }
plt.rcParams.update(params)

ndim = 13

# Chains - data loading
chains = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

labels = [r"$T_{eff}^A$ [K]",r"$T_{eff}^B$ [K]",r"$R_A [R_{\odot}]$",r"$R_B [R_{\odot}]$",r"$i [^{\circ}]$", r"$S_\mathrm{A}$", r"$S_\mathrm{A}$", r"$M_A [M_{\odot}]$",r"$M_B [M_{\odot}]$",r"$e$",r"$\omega [^{\circ}]$",r"$\gamma$ [km/s]", r"$T_0$ [d]"]
plt.rc('xtick', labelsize=6) 
plt.rc('ytick', labelsize=6)
fig, ax = plt.subplots(nrows=ndim,ncols=ndim,figsize=(17,17),dpi=300)
figure  = corner.corner(chains.T, show_titles=True, labels=labels, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], 
          label_kwargs={"fontsize": 15, "labelpad": 7}, title_kwargs={"fontsize": 12, "pad": 10}, max_n_ticks=4)                                      

ax[0,0].set_title(r'$T_\mathrm{A}$',fontsize=11,pad=10)
ax[1,1].set_title(r'$T_\mathrm{B}$',fontsize=11,pad=10)
ax[2,2].set_title(r'$R_\mathrm{A}$',fontsize=11,pad=10)
ax[3,3].set_title(r'$R_\mathrm{B}$',fontsize=11,pad=10)
ax[4,4].set_title(r'$i$',fontsize=11,pad=10)
ax[5,5].set_title(r'$S_\mathrm{A}$',fontsize=11,pad=10)
ax[6,6].set_title(r'$S_\mathrm{B}$',fontsize=11,pad=10)
ax[7,7].set_title(r'$M_\mathrm{A}$',fontsize=11,pad=10)
ax[8,8].set_title(r'$M_\mathrm{B}$',fontsize=11,pad=10)
ax[9,9].set_title(r'$e$',fontsize=11,pad=10)
ax[10,10].set_title(r'$\omega$',fontsize=11,pad=10)
ax[11,11].set_title(r'$\gamma$',fontsize=11,pad=10)
ax[12,12].set_title(r'$T_0$',fontsize=11,pad=10)

plt.subplots_adjust(wspace=0.04, hspace=0.04)
plt.tight_layout()
plt.savefig('cornerplot.pdf', dpi=300)
plt.show()


