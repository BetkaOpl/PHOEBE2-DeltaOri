#!/usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
from myphoebe import *

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

#  theta = myphoebe.initial_parameters()
#  print('theta = ', theta)

  '''
  f = open("initial_parameters.tmp", "r")
  s = f.readline()
  f.close()
  theta = list(map(lambda x: float(x), s.split()))
  print('theta = ', theta)
  '''

  theta = np.loadtxt('theta_maxprob.csv')
  print(theta)

  myphoebe.chi2(theta)

  myphoebe.plot_forward_model_phase_com('chi2_test.pdf')

if __name__ == "__main__":
  main()


