#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
from myphoebe_Tfix_l3fix import *
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

  # někdy NAN, nefunguje s titulkem
  #theta = myphoebe.initial_parameters()
  #print('Initial theta: ', theta)
  #myphoebe.chi2(theta)

  #myphoebe.plot_forward_model_phase_com('Initial_plot.pdf', r'$\chi^2$ = {} (initial)'.format(round(myphoebe.chi2(theta)),0))
  
  theta2 = np.loadtxt('best_fit.csv')
  print('PRINT')
  theta2 = theta2[1:]
  print(theta2)
  print('Final theta: ', theta2)

  myphoebe.chi2(theta2)

  myphoebe.plot_forward_model_phase_com('Final_plot.pdf', 0)
  # r'$\chi^2$ = {} (final)'.format(round(myphoebe.chi2(theta2))),
  

if __name__ == "__main__":
  main()


