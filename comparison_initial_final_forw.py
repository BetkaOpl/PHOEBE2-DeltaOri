#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
from myphoebe import *
from matplotlib.ticker import (MultipleLocator)

# Configuration file
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('setting.cfg')

import test_comp_phase
from test_comp_phase import Myphoebe2


def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe2()

  theta = myphoebe.initial_parameters()
  print('Initial theta: ', theta)
  myphoebe.chi2(theta)

  myphoebe.plot_forward_model_phase_com('Initial_plot.png', r'$\chi^2$ - initial')
  
  theta2 = np.loadtxt('theta_maxprob.csv')
  print('Final theta: ', theta2)

  myphoebe.chi2(theta2)

  myphoebe.plot_forward_model_phase_com('Final_plot.png', r'$\chi^2$ - final')

if __name__ == "__main__":
  main()


