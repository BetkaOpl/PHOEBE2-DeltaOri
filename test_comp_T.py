#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
from myphoebe import *
from matplotlib.ticker import (FormatStrFormatter)

# Configuration file
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('setting.cfg')

params = {
        'text.usetex' : True,
        'font.size'   : cfg._sections['general'].get('text_rest'),
        'font.family' : 'DejaVu Sans',
        'xtick.minor.visible' : True,
        }
plt.rcParams.update(params)


class Myphoebe2(Myphoebe):
  '''
  A minor modification of Myphoebe (inherited form).

  '''

  def chi2_contrib(self, ids, dataset):
    '''
    Print chi2 for each dataset.

    '''
    
    print('chi2 from dataset', dataset, '=', "{:19.10f}".format(np.sum(self.chi[ids])))


  def plot_forward_model(self, output='test_comp_T.pdf', dpi=600):
    '''
    Plot model from PHOEBE 2 variables.

    '''

    fig, (ax1, ax2) = plt.subplots(figsize = (10,10), nrows=2, ncols=1)

    fig.suptitle(r'$\chi^2$', fontsize=cfg._sections['general'].get('text_title'))

    s         = float(cfg._sections['styles'].get('marker_size'))
    lw        = float(cfg._sections['styles'].get('syn_lw'))
    res_lw    = float(cfg._sections['styles'].get('res_lw'))
    cs        = float(cfg._sections['styles'].get('capsize'))
    marker    = cfg._sections['styles'].get('marker_type')
    capsize   = float(cfg._sections['styles'].get('capsize'))
    err_thick = float(cfg._sections['styles'].get('err_thick'))

    ids = np.where(self.dataset==1)

    ax1.errorbar(self.x[ids]-2400000, self.yobs[ids], self.yerr[ids], color=cfg._sections['colors'].get('obs_blue'), marker=marker, lw=lw, fmt='none', label='BRITE blue - observed', capsize=capsize, elinewidth=err_thick, capthick=err_thick, zorder = 3)
    ax1.scatter(self.x[ids]-2400000, self.ysyn[ids], color=cfg._sections['colors'].get('syn'), marker=marker, s=s, lw=lw, label='BRITE - sythetic', zorder = 3)
    for i in ids:
      ax1.plot([self.x[i]-2400000, self.x[i]-2400000], [self.yobs[i], self.ysyn[i]], color=cfg._sections['colors'].get('b_rv1'), lw=res_lw, zorder = 4)

    self.chi2_contrib(ids, 1)
    
    max_lcB_T, min_lcB_T = max(self.x[ids]-2400000), min(self.x[ids]-2400000)
    max_lcB_F, min_lcB_F = max(np.r_[self.yobs[ids], self.ysyn[ids]]), min(np.r_[self.yobs[ids], self.ysyn[ids]])

    ids = np.where(self.dataset==2)
    ax1.errorbar(self.x[ids]-2400000, self.yobs[ids], self.yerr[ids], color=cfg._sections['colors'].get('obs_red'), marker=marker, lw=lw, fmt='none', label='BRITE red - observed', capsize=capsize, elinewidth=err_thick, capthick=err_thick, zorder = 3)
    ax1.scatter(self.x[ids]-2400000, self.ysyn[ids], color=cfg._sections['colors'].get('syn'), marker=marker, s=s, lw=lw, zorder = 3)
    for i in ids:
      ax1.plot([self.x[i]-2400000, self.x[i]-2400000], [self.yobs[i], self.ysyn[i]], color=cfg._sections['colors'].get('r_rv2'), lw=res_lw, zorder = 4)

    ax1.set_xlabel(r'$t$ [HJD-2400000]', labelpad=10)
    ax1.set_ylabel(r'$F$ [1]', labelpad=15)
    ax1.legend(loc='lower left', ncol=4, fontsize=cfg._sections['general'].get('text_legend'))

    self.chi2_contrib(ids, 2)

    max_lcR_T, min_lcR_T = max(self.x[ids]-2400000), min(self.x[ids]-2400000)
    max_lcR_F, min_lcR_F = max(np.r_[self.yobs[ids], self.ysyn[ids]]), min(np.r_[self.yobs[ids], self.ysyn[ids]])

    # marigns
    min_T = min(min_lcB_T, min_lcR_T)
    max_T = max(max_lcB_T, max_lcR_T)
    min_F = min(min_lcB_F, min_lcR_F)
    max_F = max(max_lcB_F, max_lcR_F)
    ax1.set_xlim(min_T-0.04*(max_T-min_T), max_T+0.04*(max_T-min_T))
    ax1.set_ylim(min_F-0.17*(max_F-min_F), max_F+0.04*(max_F-min_F))


    ids = np.where(self.dataset==3)
    ax2.errorbar(self.x[ids]-2400000, self.yobs[ids], self.yerr[ids], color=cfg._sections['colors'].get('obs_rv1'), marker=marker, lw=lw, fmt='none', label=r'RV$_1$ - observed', capsize=capsize, elinewidth=err_thick, capthick=err_thick, zorder = 3)
    ax2.scatter(self.x[ids]-2400000, self.ysyn[ids], color=cfg._sections['colors'].get('syn'), marker=marker, s=s, lw=lw, label=r'RV - sythetic', zorder = 3)
    for i in ids:
      ax2.plot([self.x[i]-2400000, self.x[i]-2400000], [self.yobs[i], self.ysyn[i]], color=cfg._sections['colors'].get('b_rv1'), lw=1, zorder = 4)
     
    self.chi2_contrib(ids, 3)   

    ids = np.where(self.dataset==4)
    ax2.errorbar(self.x[ids]-2400000, self.yobs[ids], self.yerr[ids], color=cfg._sections['colors'].get('obs_rv2'), marker=marker, lw=lw, fmt='none', label=r'RV$_2$ - observed', capsize=capsize, elinewidth=err_thick, capthick=err_thick, zorder = 3)
    ax2.scatter(self.x[ids]-2400000, self.ysyn[ids], color=cfg._sections['colors'].get('syn'), marker=marker, s=s, lw=lw, zorder = 3)
    for i in ids:
      ax2.plot([self.x[i]-2400000, self.x[i]-2400000], [self.yobs[i], self.ysyn[i]], color=cfg._sections['colors'].get('r_rv2'), lw=1, zorder = 4)

    self.chi2_contrib(ids, 4)
    
    ax2.plot([53800,59000], [0, 0], c = cfg._sections['colors'].get('rv0'), lw=2*lw, zorder=2)

    ax2.set_xlabel(r'$t$ [HJD-2400000]', labelpad=10)
    ax2.set_ylabel(r'RV [km/s]', labelpad=10)
    ax2.legend(loc='upper left', ncol=4, fontsize=cfg._sections['general'].get('text_legend'))
    ax2.set_xlim(53800, 58900)
    ax2.set_ylim(-400, 400)

    ax1.grid(linestyle="-", color='gray', lw=0.8*lw, zorder = 0)
    ax2.grid(linestyle="-", color='gray', lw=0.8*lw, zorder = 0)
    
    # right y-axis
    ax3 = ax1.twinx()
    ax4 = ax2.twinx()
    ax3.set_ylim(min_F-0.17*(max_F-min_F), max_F+0.04*(max_F-min_F))
    ax4.set_ylim(-400, 400)

    # alignment
    for tick2, tick4 in zip(ax2.yaxis.get_majorticklabels(), ax4.yaxis.get_majorticklabels()):
      tick4.set_horizontalalignment("right")
    
    # distance from y-axis 
    for tick2, tick4 in zip(ax2.get_yaxis().get_major_ticks(), ax4.get_yaxis().get_major_ticks()):
      tick4.set_pad(38)
    
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    

def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe2()

  theta = myphoebe.initial_parameters()

  myphoebe.chi2(theta)

  myphoebe.plot_forward_model()
   

if __name__ == "__main__":
  main()


