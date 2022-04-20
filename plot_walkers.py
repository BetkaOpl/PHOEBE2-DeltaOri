#!/usr/bin/python3
# MCMC - walkers 

# Imported libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_walkers(dim, p, output, burn_in, n_burn_in, dpi=600):

    # Font setting
    params = {'text.usetex' : True,
              'font.size'   : 8,
              'font.family' : 'lmodern',
              }
    plt.rcParams.update(params)



    param = [r'$T_0$ [d]', r'$T_\mathrm{A}$ [K]', r'$T_\mathrm{B}$ [K]', r'$R_\mathrm{A}$ [$R_\odot$]', r'$R_\mathrm{B}$ [$R_\odot$]', r'$i$ [°]', r'$S_\mathrm{A}$', r'$S_\mathrm{B}$', r'$M_\mathrm{A}$ [$M_\odot$]', r'$M_\mathrm{B}$ [$M_\odot$]', r'$e$', r'$\omega$ [°]', r'$\gamma$ [km/s]', r'$l_{3\mathrm{B}}$', r'$l_{\mathrm{3R}}$']



    ite = len(p[0][0])

    if dim == 13:
        fig, ((ax01, ax02, ax03, ax0), (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=4, ncols=4, figsize=(13,7))

        for ax in [ax01, ax02, ax03, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]: 
            ax.set_xlim(n_burn_in, ite+n_burn_in) 

        for ax in [ax01, ax02, ax03, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]: 
            ax.set_xticklabels([]) 

        for ax in [ax01, ax02, ax03]: 
            ax.set_yticks([])
            ax.set_xticks([]) 

        for ax in [ax9, ax10, ax11, ax12]:
            ax.set_xlabel(r'$N_\mathrm{iter}$')

        for side in ['right','top','bottom','left']:
            for ax in [ax01, ax02, ax03]:
                ax.spines[side].set_visible(False)

        ax01.scatter([1000], [50], color='white')
        ax02.scatter([1000], [50], color='white')
        ax01.text(700, 50, '{} walkers'.format(len(p[0])), fontsize=14)
        ax02.text(500, 50, '{} iterations'.format(ite), fontsize=14)
        if burn_in == False:
             ax02.text(500, 10, 'without burn-in ()', fontsize=14)

        axis = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]



    if dim == 12:
        del param[1]
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=3, ncols=4, figsize=(13,7))
        if burn_in:
            fig.suptitle('{} walkers, {} iterations'.format(len(p[0]),ite), fontsize=14)
        else:
            fig.suptitle('{} walkers, {} iterations, without burn-in'.format(len(p[0]),ite), fontsize=14)

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]: 
            ax.set_xlim(n_burn_in, ite+n_burn_in) 

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]: 
            ax.set_xticklabels([]) 

        for ax in [ax9, ax10, ax11, ax12]:
            ax.set_xlabel(r'$N_\mathrm{iter}$')

        axis = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]



    if dim == 14:
        del param[1]
        fig, ((ax01, ax02, ax00, ax0), (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=4, ncols=4, figsize=(13,7))

        for ax in [ax01, ax02, ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]: 
            ax.set_xlim(n_burn_in, ite+n_burn_in) 

        for ax in [ax01, ax02, ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]: 
            ax.set_xticklabels([]) 

        for ax in [ax01, ax02]: 
            ax.set_yticks([])
            ax.set_xticks([]) 

        for ax in [ax9, ax10, ax11, ax12]:
            ax.set_xlabel(r'$N_\mathrm{iter}$')

        for side in ['right','top','bottom','left']:
            for ax in [ax01, ax02]:
                ax.spines[side].set_visible(False)

        ax01.scatter([1000], [50], color='white')
        ax02.scatter([1000], [50], color='white')
        ax01.text(700, 50, '{} walkers'.format(len(p[0])), fontsize=14)
        ax02.text(500, 50, '{} iterations'.format(ite), fontsize=14)
        if burn_in == False:
             ax02.text(500, 10, 'without burn-in ()', fontsize=14)

        axis = [ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]


    if dim == 15:
        fig, ((ax01, ax000, ax00, ax0), (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(nrows=4, ncols=4, figsize=(13,7))

        for ax in [ax01, ax000, ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]: 
            ax.set_xlim(n_burn_in, ite+n_burn_in) 

        for ax in [ax01, ax000, ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]: 
            ax.set_xticklabels([]) 

        for ax in [ax01]: 
            ax.set_yticks([])
            ax.set_xticks([]) 

        for ax in [ax9, ax10, ax11, ax12]:
            ax.set_xlabel(r'$N_\mathrm{iter}$')

        for side in ['right','top','bottom','left']:
            for ax in [ax01]:
                ax.spines[side].set_visible(False)

        ax01.scatter([1000], [50], color='white')
        ax01.text(500, 52, '{} walkers'.format(len(p[0])), fontsize=14)
        ax01.text(500, 51, '{} iterations'.format(ite), fontsize=14)
        if burn_in == False:
             ax01.text(500, 50, 'without burn-in', fontsize=14)

        axis = [ax000, ax00, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]



    x = [*range(n_burn_in, ite+n_burn_in, 1)]

    for j, ax in zip(range(len(p)), axis):
        for i in range(len(p[j])):
            ax.plot(x, p[j][i], lw=0.5)
            ax.set_title(param[j])


    # Save the figure (.pdf)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)



if __name__ == "__main__":

    dim = 15
    n_burn_in = 400
    nwalkers = 30

    # Data loading

    p = []
    for i in range(1, dim+1):
        p.append(np.loadtxt(f'pos{i}.txt'))

    for i in range(len(p)):
        p[i] = p[i].T


    p2 = []
    for i in range(1, dim+1):
        p2.append(np.loadtxt(f'pos{i}.txt'))

    for i in range(len(p2)):
        p2[i] = p2[i][n_burn_in:]
 
    for i in range(len(p2)):
        p2[i] = p2[i].T


    plot_walkers(dim, p, 'walkers_all.pdf', True, 0)
    plot_walkers(dim, p2, 'walkers_no_burn_in.pdf', False, n_burn_in)
