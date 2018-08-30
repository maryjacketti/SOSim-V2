# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from math import *
import numpy as np
import random
import mcint
import utm
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import pdb

#bivariate equation
def CG(x,mux,sigmax):
    CG = (1.0/(np.sqrt(2.0*np.pi*(sigmax**2.0))))*np.exp((-(x-mux)**2.0)/(2.0*(sigmax**2.0)))
    return CG

#defines mu and sigma in 1-dimension
def ff(x,vx,Dx,XSpill,sigmax0,t):
    mux = XSpill + vx*t
    sigmax = sigmax0 + np.sqrt(2.0*Dx*t)
    return CG(x,mux,sigmax)

#gives the random variables for vx, Dx, and gamma to integrate over
#also starts the likelihood function calculation and the multiplication of f and the likelihood
#100,000 random values are generated for gamma1, gamma2, gamma3, gamma4 and vx and Dx to integrate over
def integ(x,xs,XSpill,ConData,sigmax0,t,SP):
	vx1 = [random.uniform(-50.0,50.0) for i in range(100000)]
	vx2 = [random.uniform(-50.0,50.0) for i in range(100000)]
	vx3 = [random.uniform(-50.0,50.0) for i in range(100000)]
	vx4 = [random.uniform(-50.0,50.0) for i in range(100000)]
	Dx1 = [random.uniform(0.01,40.1) for i in range(100000)]
	Dx2 = [random.uniform(0.01,40.1) for i in range(100000)]
	Dx3 = [random.uniform(0.01,40.1) for i in range(100000)]
	Dx4 = [random.uniform(0.01,40.1) for i in range(100000)]
	ga = [np.random.random(4) for i in range(100000)]
	gamma1 = np.zeros(len(ga))
	gamma2 = np.zeros(len(ga))
	gamma3 = np.zeros(len(ga))
	gamma4 = np.zeros(len(ga))

	for i in range(len(ga)):
	    ga[i]/=ga[i].sum()
	    gamma1[i] = ga[i][0]
	    gamma2[i] = ga[i][1]
	    gamma3[i] = ga[i][2]
	    gamma4[i] = ga[i][3]

	#starts the likelihood function calculation	    
	IniIndLikelihood = np.ones([len(xs),len(vx1)])
	Lamda = np.zeros(shape=(len(vx1)))
	Prob = np.zeros(shape=(len(vx1)))
	for ci in range(len(xs)):
		if ConData[ci] > 0.0:
			for i in range(len(vx1)):
				Prob[i] = gamma1[i]*ff(xs[ci],vx1[i],Dx1[i],XSpill,sigmax0,SP) + gamma2[i]*ff(xs[ci],vx2[i],Dx2[i],XSpill,sigmax0,SP) + gamma3[i]*ff(xs[ci],vx3[i],Dx3[i],XSpill,sigmax0,SP) + gamma4[i]*ff(xs[ci],vx4[i],Dx4[i],XSpill,sigmax0,SP)
				if Prob[i] > 1e-308:
					Lamda[i] = 1/Prob[i]
					IniIndLikelihood[ci,i] = Lamda[i]*np.exp(-Lamda[i]*ConData[ci])
				else:
					Lamda[i] = 0.0
					IniIndLikelihood[ci,i] = 0.0

	CompLikelihood = np.ones([len(vx1)])
	Likelihood = np.zeros([len(vx1)])
	for i in range(len(vx1)):
		for ci in range(len(xs)):
			if ConData[ci] > 0.0:
				if IniIndLikelihood[ci,i] == 0.0:
					CompLikelihood[i] = 0.0

	MaxLogLike = -22.0
	for i in range(len(vx1)):
		for ci in range(len(xs)):
			if ConData[ci] > 0.0:
				if CompLikelihood[i] == 1.0:
					Likelihood[i] = Likelihood[i] + np.log(IniIndLikelihood[ci,i])

		if CompLikelihood[i] == 1.0:
			if MaxLogLike == -22.0:
				MaxLogLike = Likelihood[i]
			else:
				MaxLogLike = np.max([MaxLogLike,Likelihood[i]])

	for i in range(len(vx1)):
		if CompLikelihood[i] == 1.0:
			Likelihood[i] = Likelihood[i] - MaxLogLike #+ 700.0
			Likelihood[i] = np.exp(Likelihood[i])

	#this is where I find the Mximum likelihood estimator parameters for each patch
	#it takes the index of the likelihood function array and then finds the values of the parameters at that location
	Likelihoodpos = np.argmax(Likelihood)
	print Likelihoodpos
	print vx1[Likelihoodpos], vx2[Likelihoodpos], vx3[Likelihoodpos], vx4[Likelihoodpos]
	print Dx1[Likelihoodpos], Dx2[Likelihoodpos], Dx3[Likelihoodpos], Dx4[Likelihoodpos]
	print gamma1[Likelihoodpos], gamma2[Likelihoodpos], gamma3[Likelihoodpos], gamma4[Likelihoodpos]

	#this starts the multiplication of f from the grid with the likelihood function
	#returns the result of the multiplication
	ConResult = np.zeros([len(t),len(x)])
	for ti in range(len(t)):
		for ci in range(len(x)):
			ProbObsGivenParam = np.zeros([len(vx1)])
			integral = 0.0
			for i in range(len(vx1)):
				ProbObsGivenParam[i] = gamma1[i]*ff(x[ci],vx1[i],Dx1[i],XSpill,sigmax0,t[ti]) + gamma2[i]*ff(x[ci],vx2[i],Dx2[i],XSpill,sigmax0,t[ti]) + gamma3[i]*ff(x[ci],vx3[i],Dx3[i],XSpill,sigmax0,t[ti]) + gamma4[i]*ff(x[ci],vx4[i],Dx4[i],XSpill,sigmax0,t[ti])
			integral = 0.0
			for i in range(len(vx1)):
				integral = integral + ProbObsGivenParam[i]*Likelihood[i]
			ConResult[ti,ci] = integral
	print MaxLogLike
	return ConResult


#this is where we upload the data
if __name__ == "__main__":

	sigmax0 = 0.0
	XSpill = 0.0
	t = [1.0]
	SP = 1.0
	scale = 80.
	Node = 120
	xs = [1,5,9,20,24,28]
	ConData = np.array([5,30,5,5,30,5])/100

	x = np.linspace(XSpill-scale/2,XSpill+scale,Node+1)
	Result = integ(x,xs,XSpill,ConData,sigmax0,t,SP)
	plt.plot(x,Result[0,:])
	plt.show()







