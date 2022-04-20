#!/usr/bin/env python3
# coding: utf-8

__author__ = "Alzbeta Oplitilova (betsimsim@seznam.cz)"
__version__ = "Apr 20th 2022"

import time
import numpy as np
import nlopt
import emcee

import phoebe
from phoebe import u


class Myphoebe(object):
  '''
  My wrapper for Phoebe2 computations.
  '''

  def __init__(self, debug=True):
    '''
    Initialisation (input data, Phoebe2, fixed parameters).
    '''
    self.debug    = debug

    # Input data
    tb, mb_obs, mb_err, nb     = np.loadtxt('BRITE_BLUE_1LC.dat', unpack=True, usecols=[0, 1, 2, 4])
    tr, mr_obs, mr_err, nr     = np.loadtxt('BRITE_RED_1LC.dat',  unpack=True, usecols=[0, 1, 2, 4])
    t1, rv1_obs, rv1_err, nrv1 = np.loadtxt('RV1.dat',            unpack=True, usecols=[0, 1, 2, 4])
    t2, rv2_obs, rv2_err, nrv2 = np.loadtxt('RV2.dat',            unpack=True, usecols=[0, 1, 2, 4])

    print(len(tb), len(tr), len(t1), len(t2))
    print('Number of data points: ', len(tb)+len(tr)+len(t1)+len(t2))

    fluxb_obs = 10.0**(-0.4*mb_obs)
    fluxr_obs = 10.0**(-0.4*mr_obs)
    fluxb_err = fluxb_obs*(10**(0.4*mb_err)-1)
    fluxr_err = fluxr_obs*(10**(0.4*mr_err)-1)

    # Errors multiplied by factors
    fluxb_err *= 5.0
    fluxr_err *= 5.0
    rv1_err   *= 5.0
    rv2_err   *= 5.0

    # Single vector(s)
    self.x       = np.r_[tb, tr, t1, t2]
    self.yobs    = np.r_[fluxb_obs, fluxr_obs, rv1_obs, rv2_obs]
    self.yerr    = np.r_[fluxb_err, fluxr_err, rv1_err, rv2_err]
    self.dataset = np.r_[nb, nr, nrv1, nrv2]
    self.ysyn    = None
    self.chi     = None

    #logger = phoebe.logger('error', filename='mylog.log')

    self.b = phoebe.default_binary()

    self.b.add_dataset('lc', compute_times=tb, dataset='lcB', intens_weighting='photon', passband='BRITE:blue')
    self.b.add_dataset('lc', compute_times=tr, dataset='lcR', intens_weighting='photon', passband='BRITE:red')
    self.b.add_dataset('rv', compute_times=t1, dataset='rv1', intens_weighting='photon', passband='Johnson:V')
    self.b.add_dataset('rv', compute_times=t2, dataset='rv2', intens_weighting='photon', passband='Johnson:V')

    self.b['ntriangles@primary']   = 1500
    self.b['ntriangles@secondary'] = 500

    # Fixed parameters
    self.b.set_value('period', component='binary', value=5.732436*u.d)     # Mayer et al. (2010)
    self.b.set_value('dperdt', component='binary', value=1.45*u.deg/u.yr)  # Pablo et al. (2015)

    self.b.set_value('l3_mode', 'fraction', dataset='lcB')
    self.b.set_value('l3_mode', 'fraction', dataset='lcR')

    # Other parameters
    self.b.set_value('atm', component='primary',   value='blackbody')
    self.b.set_value('atm', component='secondary', value='blackbody')

    self.b.set_value(qualifier='ld_mode', dataset='lcB', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcR', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv1', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv2', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcB', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='lcR', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv1', component='secondary', value='manual')
    self.b.set_value(qualifier='ld_mode', dataset='rv2', component='secondary', value='manual')

    self.b.set_value(qualifier='ld_func',   dataset='lcB', component='primary',   value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='lcR', component='primary',   value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='rv1', component='primary',   value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='rv2', component='primary',   value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='lcB', component='secondary', value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='lcR', component='secondary', value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='rv1', component='secondary', value='linear')
    self.b.set_value(qualifier='ld_func',   dataset='rv2', component='secondary', value='linear')

    self.b.set_value(qualifier='ld_coeffs', dataset='lcB', component='primary',   value=0.3048568945852137)  # computed by limcof for blue region
    self.b.set_value(qualifier='ld_coeffs', dataset='lcR', component='primary',   value=0.22801460197477924) # computed by limcof for red region
    self.b.set_value(qualifier='ld_coeffs', dataset='rv1', component='primary',   value=0.3048568945852137)  # computed by limcof for blue region
    self.b.set_value(qualifier='ld_coeffs', dataset='rv2', component='primary',   value=0.3048568945852137)  # computed by limcof for blue region
    self.b.set_value(qualifier='ld_coeffs', dataset='lcB', component='secondary', value=0.2995419019169233)  # computed by limcof for blue region
    self.b.set_value(qualifier='ld_coeffs', dataset='lcR', component='secondary', value=0.21435273667038388) # computed by limcof for red region
    self.b.set_value(qualifier='ld_coeffs', dataset='rv1', component='secondary', value=0.2995419019169233)  # computed by limcof for blue region
    self.b.set_value(qualifier='ld_coeffs', dataset='rv2', component='secondary', value=0.2995419019169233)  # computed by limcof for blue region

    self.b.set_value(qualifier='ld_mode_bol', component='primary',   value='manual')
    self.b.set_value(qualifier='ld_mode_bol', component='secondary', value='manual')

    self.b.set_value('gravb_bol', component='primary',   value=1.0)
    self.b.set_value('gravb_bol', component='secondary', value=1.0)

    self.b.set_value('irrad_frac_refl_bol', component='primary',   value=1.0)
    self.b.set_value('irrad_frac_refl_bol', component='secondary', value=1.0)

    self.b.flip_constraint('mass@primary',   solve_for='sma@binary')
    self.b.flip_constraint('mass@secondary', solve_for='q')

    self.b.set_value('teff', component='primary',   value=31000*u.K)

    # Save 'twigs'
    if self.debug:
      f = open('twigs.txt', 'w')
      for twig in self.b.twigs:
        f.write("%s\n" % twig)
      f.close()


  def model(self, theta):
    '''
    Synthetic fluxes from the model.

    :param theta: Vector of free parameters.
    :return:

    '''

    T0,T2,R1,R2,I,SA,SB,M1,M2,e,omega,gamma,l3B,l3R = theta
   
    self.b.set_value('ecc',        component='binary',    value=e)
    self.b.set_value('per0',       component='binary',    value=omega*u.deg)
    self.b.set_value('incl',       component='binary',    value=I*u.deg)
    self.b.set_value('t0_supconj', component='binary',    value=T0*u.d)
    self.b.set_value('teff',       component='secondary', value=T2*u.K)
    self.b.set_value('requiv',     component='primary',   value=R1*u.solRad)
    self.b.set_value('requiv',     component='secondary', value=R2*u.solRad)
    self.b.set_value('mass',       component='primary',   value=M1*u.solMass)
    self.b.set_value('mass',       component='secondary', value=M2*u.solMass)
    self.b.set_value('vgamma',                            value=gamma*u.km/u.second)
    self.b.set_value('l3_frac', dataset='lcB', value=l3B)  
    self.b.set_value('l3_frac', dataset='lcR', value=l3R)

    self.b.run_compute(distortion_method='roche', irrad_method='wilson', ltte=False, rv_method='flux-weighted', rv_grav=False)

    #print('log g: ', self.b.filter('logg', context='component'))

    fluxb_syn = self.b['fluxes@lcB@latest@model'].value
    fluxr_syn = self.b['fluxes@lcR@latest@model'].value
    rv1_syn   = self.b['rvs@primary@rv1@latest@model'].value
    rv2_syn   = self.b['rvs@secondary@rv2@latest@model'].value

    # Normalisation
    fluxb_nor = SB*fluxb_syn/np.amax(fluxb_syn)
    fluxr_nor = SA*fluxr_syn/np.amax(fluxr_syn)

    # Save model
    if self.debug:
      self.b.save('forward_model.phoebe')
      f = open('forward_model.txt', 'w')
      np.savetxt(f, np.column_stack((self.b['times@lcB@latest@model'].value, fluxb_nor)),         fmt='%22.16f', header="times,fluxb_nor")
      np.savetxt(f, np.column_stack((self.b['times@lcR@latest@model'].value, fluxr_nor)),         fmt='%22.16f', header="times,fluxr_nor")
      np.savetxt(f, np.column_stack((self.b['times@primary@rv1@latest@model'].value, rv1_syn)),   fmt='%22.16f', header="times,rv1_syn")
      np.savetxt(f, np.column_stack((self.b['times@secondary@rv2@latest@model'].value, rv2_syn)), fmt='%22.16f', header="times,rv2_syn")
      f.close()

    return np.r_[fluxb_nor, fluxr_nor, rv1_syn, rv2_syn]

  def chi2(self, theta):
    '''
    Computes chi^2.

    :param theta: Vector of free parameters.
    :return:

    '''
    if self.debug:
      print('theta = ', theta)

    self.ysyn = self.model(theta)
    self.chi  = ((self.yobs - self.ysyn)/self.yerr)**2
    chi_sum   = np.sum(self.chi)

    if np.isnan(chi_sum):
      ids = np.where(~np.isnan(self.chi))
      chi_sum = np.sum(self.chi[ids])
      print('Warning!! self.chi = nan; number_of_nans = ', len(self.chi)-len(ids[0]))

    if self.debug:
      print('chi_sum = ', chi_sum)
      print('CHI_DEBUG')

      ids = np.where(~np.isnan(self.chi))
      f = open(f'chi2_func.tmp', 'a')
      f.write("%22.16f %d " % (chi_sum, len(ids[0])))
      for tmp in theta: f.write(" %22.16f" % tmp)
      f.write("\n")
      f.close()

      f = open(f'obs_syn_chi.dat', 'a+')
      f.seek(0)
      it = (sum(1 for _ in f))/314 + 1 
      f.write("x                            yobs                   ysyn                   chi                 iteration: %d\n" % it)
      for i in range(len(self.x)): f.write("%22.16f %22.16f %22.16f %22.16f\n" % (self.x[i], self.yobs[i], self.ysyn[i], self.chi[i]))
      f.write("\n")
      f.close()

    return chi_sum

  def lnlike(self, theta):
    '''
    Likelihood +ln p(data|theta).

    :param theta: Vector of free parameters.
    :return:

    '''
    self.ysyn = self.model(theta)
    self.chi  = ((self.yobs - self.ysyn)/self.yerr)**2
    ids       = np.where(~np.isnan(self.chi))
    lp        = -0.5*np.sum(self.chi[ids] + np.log(self.yerr[ids]**2) + np.log(2.0*np.pi))

    if self.debug:
      chi_sum = np.sum(self.chi[ids])
      print('chi_sum = ', chi_sum, ' lp = ', lp)
      f = open(f'chi2_func.tmp', 'a')
      f.write("%16.8f %d " % (chi_sum, len(ids[0])))
      for tmp in theta: f.write(" %22.16f" % tmp)
      f.write("\n")
      f.close()

    return lp


  def lnprob(self, theta):
    '''
    Posterior ln p(theta|data).

    :param theta: Vector of free parameters.
    :return:

    '''
    lp = self.lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + self.lnlike(theta)

  def initial_parameters(self):
    '''
    Setting of initial parameters

    :return theta: Vector of free parameters.

    '''

    T0    = 2457733.824876185507     # d
    T2    = 20851.20634158405664     # K
    R1    = 13.50883415032292945     # R_Sol
    R2    = 3.825274099025570251     # R_Sol
    I     = 77.61956864686750635     # deg
    SA    = 1.023568130179173252     # 1
    SB    = 1.024848843136050736     # 1
    M1    = 19.51535083586710329     # M_Sol
    M2    = 8.905782406623607983     # M_Sol
    e     = 0.081127844206919833     # 1
    omega = 139.9798807596344261     # deg
    gamma = 0.01330919526709171963   # km/s
    l3B   = 0.273
    l3R   = 0.273

    theta = T0,T2,R1,R2,I,SA,SB,M1,M2,e,omega,gamma,l3B,l3R

    return theta


  def lnprior(self, theta):
    '''
    Prior +ln p(theta). Uninformative; assures appropriate ranges.
    Note: without a normalisation of p!

    :param theta: Vector of free parameters.
    :return:

    '''

    T0,T2,R1,R2,I,SA,SB,M1,M2,e,omega,gamma,l3B,l3R = theta
        
    if  2457733.8493-0.2 < T0 < 2457733.8493+0.2 and \
                   15000 < T2    < 30000 and \
                   10    < R1    < 20    and \
                   2     < R2    < 10    and \
                   65    < I     < 89    and \
                   0.9   < SA    < 1.2   and \
                   0.9   < SB    < 1.2   and \
                   15    < M1    < 35    and \
                   3     < M2    < 20    and \
                   0     < e     < 0.2   and \
                   90    < omega < 180   and \
                   -5    < gamma < 35    and \
                   0.15  < l3B   < 0.4   and \
                   0.15  < l3R   < 0.4:
        return 0.0
    else:
        return -np.inf



  def lower_bounds(self):
    '''
    Lower bounds for nlopt.

    :return theta: Vector of free parameters.

    '''
    T0    = 2457733.8493-0.2
    T1    = 25000
    T2    = 20000
    R1    = 10
    R2    = 2
    I     = 65
    SA    = 0.9
    SB    = 0.9
    M1    = 18
    M2    = 3
    e     = 0
    omega = 90
    gamma = 0
    l3B   = 0.15
    l3R   = 0.15

    theta = T0,T2,R1,R2,I,SA,SB,M1,M2,e,omega,gamma,l3B,l3R

    return theta


  def upper_bounds(self):
    '''
    Upper bounds for nlopt.

    :return theta: Vector of free parameters.

    '''
    T0    = 2457733.8493+0.2
    T1    = 30000
    T2    = 30000
    R1    = 20
    R2    = 10
    I     = 89
    SA    = 1.1
    SB    = 1.1
    M1    = 35
    M2    = 20
    e     = 0.2
    omega = 180
    gamma = 35
    l3B   = 0.4
    l3R   = 0.4

    theta = T0,T2,R1,R2,I,SA,SB,M1,M2,e,omega,gamma,l3B,l3R

    return theta

def run_nlopt(myphoebe, algorithm=nlopt.LN_SBPLX, ftol=1e-6, maxeval=1000):
  '''

  Run optimisation.

  :param myphoebe: Ref. to myphoebe object.
  :param algorithm: Algorithm, e.g., nlopt.LN_NELDERMEAD, nlopt.LN_SBPLX, ...
  :param ftol: Tolerance to stop.
  :param maxeval: Maximum number of evaluations.
  :return:

  '''

  def myfunc(theta, grad):
    return myphoebe.chi2(theta)

  theta = myphoebe.initial_parameters()

  dim = len(theta)
  opt = nlopt.opt(algorithm, dim)

  print('Number of dimensions:', opt.get_dimension())
  print('Algorithm:', opt.get_algorithm_name())

  opt.set_lower_bounds(myphoebe.lower_bounds())
  opt.set_upper_bounds(myphoebe.upper_bounds())

  opt.set_ftol_rel(ftol)
  opt.set_maxeval(maxeval)
  opt.set_min_objective(myfunc)

  best_fit_theta = opt.optimize(theta)
  best_fit_chi2 = opt.last_optimum_value()
  print('Result code: ', opt.last_optimize_result())

  np.savetxt('best_fit.csv', np.r_[best_fit_chi2,best_fit_theta], delimiter=',', header='best_fit_chi2,best_fit_theta')

  print('run_nlopt() has ended sucessfully!')


def p0_func(theta, nwalkers=None, delta=0.05):
  '''

  Creating initial positions of walkers.

  :param theta: Vector of free parameters.
  :param nwalkers: Number of walkers.
  :param delta: Dispesion of random numbers.
  :return:

  '''
  p0 = []
  for i in range(nwalkers):
    tmp = []
    for j in range(len(theta)):
      if j == 0:
        tmp.append(np.random.uniform(theta[j]-0.001, theta[j]+0.001))  # T0
      else:
        tmp.append(np.random.uniform(theta[j]*(1.0-delta), theta[j]*(1.0+delta)))
    p0.append(tmp)

  return np.array(p0)


def run_mcmc(myphoebe, nwalkers=30, niter=2000, seed=1, thin=1, **kwarg):
  '''
  Running Monte-Carlo-Markov-Chain method.

  :param myphoebe: Ref. to myphoebe object.
  :param nwalkers: Number of walkers; minimum is 2 times the number of free parameters.
  :param niter: Number of iterations.
  :param seed: Random seed.
  :param thin: Use only every thin step from the chain.
  :return:

  '''
  theta = myphoebe.initial_parameters()
  print('theta = ', theta)

  np.random.seed(seed)
  p0 = p0_func(theta, nwalkers=nwalkers)

  print('Checking p0:')
  for tmp in p0:
    lp = myphoebe.lnprior(tmp)
    print('lp = ', lp)
    if not np.isfinite(lp):
      print('theta = ', np.array(tmp))
      raise ValueError('p0 out of range in self.lnprior()')

  ndim = len(theta)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, myphoebe.lnprob, **kwarg)

  print("Running production...")
  t1 = time.time()

  pos, prob, state = sampler.run_mcmc(p0, 1, progress=True)

  for i in range(niter):
    print('iter = ', i)

    pos, prob, state = sampler.run_mcmc(None, 1, progress=True)

    with open(f'chain.tmp', 'a') as f:
      for j in range(0,len(pos)):
        f.write("%d %d" % (i, j))
        for k in range(0,len(pos[j])):
          f.write(" %22.16f" % (pos[j][k]))
        f.write("\n")

    with open(f'prob.tmp', 'a') as f:
      for j in range(0,len(prob)):
        f.write("%3d %3d %16.8f\n" % (i, j, prob[j]))

    k = 0
    for tmp in pos.T:
      if k < len(pos[0]):
        k += 1
      with open(f'pos{k}.txt', 'a') as f:
        f.write("\n")
        np.savetxt(f, tmp, fmt='%22.16f', newline=' ', delimiter='')

  print("Average acceptance fraction:", np.around(np.mean(sampler.acceptance_fraction),3), "(it should be between 0.2-0.5)")
  try:
     print("Autocorrelation time estimate:", sampler.get_autocorr_time(), "(it should be around n x 10)")

  except NameError:
     print("Warning: Autocorrelation time can not be reliably estimated!")

  samples = sampler.flatchain
  theta_maxprob = samples[np.argmax(sampler.flatlnprobability)]
  chain = sampler.get_chain(thin=thin, flat=True, discard=0)

  np.savetxt('theta_maxprob.csv', theta_maxprob, delimiter=',')
  np.savetxt('chain.csv', chain, delimiter=',')

  t2 = time.time()
  print("Time: ", t2-t1, " s = ", (t2-t1)/3600.0, " h")

  print('run_mcmc() has ended sucessfully!')



def main():
  '''
  Main program.

  '''

  myphoebe = Myphoebe()

  theta = myphoebe.initial_parameters()

  myphoebe.model(theta)
  myphoebe.chi2(theta)

  #run_nlopt(myphoebe)

  run_mcmc(myphoebe)

#  print(vars(myphoebe))  # dbg

if __name__ == "__main__":
  main()
