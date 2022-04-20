#!/usr/bin/python3
# MCMC - corner plot - from all iterations and without burn-ins

# Imported libraries
import corner
import numpy as np
import matplotlib.pyplot as plt


def corner_plot(chains, output, ndim, truths, dpi=600):

    # Font setting
    params = {'text.usetex' : True,
            'font.size'   : 8,
             'font.family' : 'lmodern',
             }
    plt.rcParams.update(params)


    labels_plain = ["T0 [HJD]", "Teff_A [K]", "Teff_B [K]", "R_A [RS]", "R_B [RS]", "i [°]", "S_A", "S_B", "M_A [MS]", "M_B [MS]", "e", "omega [°]", "gamma [km/s]", "l3B", "l3R"]

    labels = [r"$T_0$ [d]", r"$T_{\mathrm{eff}}^A$ [K]", r"$T_{\mathrm{eff}}^B$ [K]", r"$R_\mathrm{A} [\hbox{$\mathcal{R}^{\rm N}_\odot$}]$", r"$R_\mathrm{B} [\hbox{$\mathcal{R}^{\rm N}_\odot$}]$", r"$i [^{\circ}]$", r"$S_\mathrm{A}$", r"$S_\mathrm{B}$", r"$M_\mathrm{A} [\hbox{$\mathcal{M}^{\rm N}_\odot$}]$",r"$M_\mathrm{B} [\hbox{$\mathcal{M}^{\rm N}_\odot$}]$",r"$e$",r"$\omega [^{\circ}]$",r"$\gamma$ [km/s]",r"$l_{3\mathrm{B}}$",r"$l_{3\mathrm{R}}$"]

    fig, ax = plt.subplots(nrows = ndim, ncols = ndim, figsize=(17,17), dpi=300)                                

    if ndim==13:
        c = 0
        ax[1,1].set_title(r'$T_\mathrm{A}$', fontsize=11, pad=10)

    if ndim==12:
        c = 1
        del labels[1]
        del labels_plain[1]

    if ndim==14:
        c = 1
        del labels[1]
        del labels_plain[1]
        ax[12,12].set_title(r'$l_{3\mathrm{B}}$', fontsize=11, pad=10)
        ax[13,13].set_title(r'$l_{3\mathrm{R}}$', fontsize=11, pad=10)

    if ndim==15:
        c = 0
        ax[13,13].set_title(r'$l_{3\mathrm{B}}$', fontsize=11, pad=10)
        ax[14,14].set_title(r'$l_{3\mathrm{R}}$', fontsize=11, pad=10)

    print("Most probable solution:\n")
    for i in range(ndim):
        print(labels_plain[i].rjust(14, ' '), ' = ', truths[i],'\n')


    ax[0,0].set_title(r'$T_0$', fontsize=11, pad=10)
    ax[2-c,2-c].set_title(r'$T_\mathrm{B}$', fontsize=11, pad=10)
    ax[3-c,3-c].set_title(r'$R_\mathrm{A}$', fontsize=11, pad=10)
    ax[4-c,4-c].set_title(r'$R_\mathrm{B}$', fontsize=11, pad=10)
    ax[5-c,5-c].set_title(r'$i$ [°]', fontsize=11, pad=10)
    ax[6-c,6-c].set_title(r'$S_\mathrm{A}$', fontsize=11, pad=10)
    ax[7-c,7-c].set_title(r'$S_\mathrm{B}$', fontsize=11, pad=10)
    ax[8-c,8-c].set_title(r'$M_\mathrm{A}$', fontsize=11, pad=10)
    ax[9-c,9-c].set_title(r'$M_\mathrm{B}$', fontsize=11, pad=10)
    ax[10-c,10-c].set_title(r'$e$', fontsize=11, pad=10)
    ax[11-c,11-c].set_title(r'$\omega$', fontsize=11, pad=10)
    ax[12-c,12-c].set_title(r'$\gamma$', fontsize=11, pad=10)

    plt.rc('xtick', labelsize=6) 
    plt.rc('ytick', labelsize=6)

    figure  = corner.corner(chains.T, show_titles=True, labels=labels, plot_datapoints=True,  label_kwargs={"fontsize": 15, "labelpad": 7}, title_kwargs={"fontsize": 12, "pad": 10}, max_n_ticks=4)

    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.tight_layout()

    plt.savefig(output, dpi=dpi)



if __name__ == "__main__":
    truths = np.loadtxt('theta_maxprob.csv')

    ndim = len(truths)
    burn_in = 300
    nwalkers = 30

    # Chains - data loading
    if ndim == 12:
        chains = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        chains2 = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], skiprows = burn_in*nwalkers)
    if ndim == 13:
        chains = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        chains2 = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], skiprows = burn_in*nwalkers)
    if ndim == 14:
        chains = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        chains2 = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], skiprows = burn_in*nwalkers)
    if ndim == 15:
        chains = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        chains2 = np.loadtxt('chain.tmp', unpack=True, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], skiprows = burn_in*nwalkers)

    corner_plot(chains,  output='cornerplot_all.pdf',        ndim=ndim, truths=truths, dpi=600)
    corner_plot(chains2, output='cornerplot_no_burn-in.pdf', ndim=ndim, truths=truths, dpi=600)
