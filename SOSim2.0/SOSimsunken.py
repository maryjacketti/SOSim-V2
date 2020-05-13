# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from math import *
import random
import utm
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import integrate
from scipy.stats import expon, norm
from scipy.stats import chi2
import pandas as pd
import datetime
from multiprocessing import pool
import multiprocessing as mp
from multiprocessing import cpu_count
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
import oceansdb
import numpy as np
import math, sys, time 
import scipy.optimize as optimize
from  scipy.optimize import differential_evolution
import cv2
import geopy.distance
import netCDF4 as nc4 

ourinformation = {}
parameter = []

def fftv(x,y,mux,muy,Dx,Dy,ro):
    s=1./(2*np.pi*Dx*Dy*np.sqrt(1-ro**2))*np.exp(-1/(2*(1-ro**2))*((x-mux)**2/Dx**2+(y-muy)**2/Dy**2-2*ro*(x-mux)*(y-muy)/(Dx*Dy)))
    return s 

def B_sampling(DLx,DLy,mux,muy,sigmax,sigmay,ro): #Q2
    """Definition of term B in the Bivariate normal Gaussian Distribution using the sampling points"""
    Bs=((((DLx-mux))**(2.0))/((sigmax)**2.0))+((((DLy-muy))**(2.0))/((sigmay)**2.0))-((2.0*(ro)*(DLx-mux)*(DLy-muy))/(sigmax*sigmay))
    return Bs

def CG(sigmax,sigmay,BuSamp,ro):
    """Conditional Gaussian function:"""
    CG=((1.0)/(2.0*(np.pi)*sigmax*sigmay*(np.sqrt(1-((ro)**2.0)))))*(np.exp(-(BuSamp)/(2.0*(1.0-((ro)**2.0)))))
    return CG

# ff defines the gaussian function
def ff(x,y,vx,vy,Dx,Dy,ro,x0,y0,Dx0,Dy0,dt,t,s):
    k=0
    for i in range(len(s)):
        mux = x0 + vx*(t-s[i])
        muy = y0 + vy*(t-s[i])
        sigmax = np.sqrt(2.0*Dx*(dt-s[i]))+Dx0
        sigmay = np.sqrt(2.0*Dy*(dt-s[i]))+Dy0
        mm = CG(sigmax,sigmay,B_sampling(x,y,mux,muy,sigmax,sigmay,ro),ro)
        k = k + mm/len(s)
    return k

def ffcont(x,y,vx,vy,Dx,Dy,ro,x0,y0,Dx0,Dy0,dt,t,s):
    k=0
    for i in range(len(s)):
        mux = x0[i] + vx*(t[i]-s[i])
        muy = y0[i] + vy*(t[i]-s[i])
        sigmax = np.sqrt(2.0*Dx*(dt[i]-s[i]))+Dx0[i]
        sigmay = np.sqrt(2.0*Dy*(dt[i]-s[i]))+Dy0[i]
        mm = CG(sigmax,sigmay,B_sampling(x,y,mux,muy,sigmax,sigmay,ro),ro)
        k = k + mm/len(s)
    return k

def CalTime(a,b):
    start = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    ends = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    diff = ends - start
    return diff.total_seconds()/86400.

def sampler(N):
    print ourinformation
    vx_min = float(ourinformation['vxmin']) # user's inputs vx_min
    vx_max = float(ourinformation['vxmax']) # user's input vx_max
    vy_min = float(ourinformation['vymin']) # user's inputs vy_min
    vy_max = float(ourinformation['vymax']) # user's inputs vy_max

    Dx_min = float(ourinformation['dxmin'])  # user's inputs Dx_min
    Dx_max = float(ourinformation['dxmax'])   # user's inputs Dx_max
    Dy_min = float(ourinformation['dymin'])  # user's inputs Dy_min
    Dy_max = float(ourinformation['dymax'])   # user's inputs Dy_min

    ro_min = -0.999
    ro_max = 0.999

    gamma_min = 0.00
    gamma_max = 1.0

    if ourinformation['SpillPlace'] == 'River':
        seed = N
    if ourinformation['SpillPlace'] == 'Ocean':
        seed = 100000


    random.seed(seed)
    vx1 = [random.uniform(vx_min,vx_max) for i in range(N)]
    vx2 = [random.uniform(vx_min,vx_max) for i in range(N)]
    vx3 = [random.uniform(vx_min,vx_max) for i in range(N)]
    vx4 = [random.uniform(vx_min,vx_max) for i in range(N)]
    vy1 = [random.uniform(vy_min,vy_max) for i in range(N)]
    vy2 = [random.uniform(vy_min,vy_max) for i in range(N)]
    vy3 = [random.uniform(vy_min,vy_max) for i in range(N)]
    vy4 = [random.uniform(vy_min,vy_max) for i in range(N)]
    Dx1 = [random.uniform(Dx_min,Dx_max) for j in range(N)]
    Dx2 = [random.uniform(Dx_min,Dx_max) for j in range(N)]
    Dx3 = [random.uniform(Dx_min,Dx_max) for j in range(N)]
    Dx4 = [random.uniform(Dx_min,Dx_max) for j in range(N)]
    Dy1 = [random.uniform(Dy_min,Dy_max) for j in range(N)]
    Dy2 = [random.uniform(Dy_min,Dy_max) for j in range(N)]
    Dy3 = [random.uniform(Dy_min,Dy_max) for j in range(N)]
    Dy4 = [random.uniform(Dy_min,Dy_max) for j in range(N)]
    ro1 = [random.uniform(ro_min,ro_max) for j in range(N)]
    ro2 = [random.uniform(ro_min,ro_max) for j in range(N)]
    ro3 = [random.uniform(ro_min,ro_max) for j in range(N)]
    ro4 = [random.uniform(ro_min,ro_max) for j in range(N)]

    np.random.seed(seed)
    ga = [np.random.random(4) for i in range(N)]
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
    return zip(vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4)

def IniLikelihood(a,parameter):
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    DLx=a.DLxx
    DLy=a.DLyy
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04
    DLcon=a.DLconny
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    st=np.array([a.st]*len(DLx)) 
  
    dt = st
    ss1 = a.ss11
    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0

    CompLikelihood=1
    
    for ci in range(len(DLx)):
        Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,dt[ci],st[ci],ss1[ci]) \
         +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,dt[ci],st[ci],ss1[ci]) \
         +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,dt[ci],st[ci],ss1[ci]) \
         +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,dt[ci],st[ci],ss1[ci])
        if Prob>1e-308:
            Lamda = 1/Prob                
            IniIndLikelihood[ci] = np.log(Lamda)-(Lamda*DLcon[ci])
        else:
            Lamda = 0
            IniIndLikelihood[ci] = 0  
          

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    return IniIndLikelihood

#___________________________confidence bounds___________________________________________

def IniLikelihood1(a,parameter):
    vx1,vx2,vx3,vx4 = parameter
    Max = a.MaxLogLikeP
    r = a.rr
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19]
    vy1 = r[4]
    vy2 = r[5]
    vy3 = r[6]
    vy4 = r[7]    
    Dx1 = r[8]
    Dx2 = r[9]
    Dx3 = r[10]
    Dx4 = r[11]        
    Dy1 = r[12]
    Dy2 = r[13]
    Dy3 = r[14]
    Dy4 = r[15]
    xx = a.xx
    DLx = a.DLxx
    yy = a.yy
    DLy = a.DLyy
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04

    DLcon = a.DLconny
    Cprior = a.Cprior
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    ptt = np.array([a.pt]*len(a.xx))
    pt = np.array([a.pt]*len(a.xx))
    st=np.array([a.st]*len(DLx))
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)

    s1 = a.ss1_tcf
    s3 = ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,st[ci],st[ci],s1[ci]) \
                +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,st[ci],st[ci],s1[ci]) \
                +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,st[ci],st[ci],s1[ci]) \
                +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,st[ci],st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    IniLikF = np.sum(IniIndLikelihood)

    IniIndLikelihoodP = np.ones([len(xx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(xx)):
        if Cprior[ci] >0:
            for i in range(1):
                ProbP = gamma1*ff(xx[ci],yy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,pt[ci],pt[ci],s3[ci]) \
                +gamma2*ff(xx[ci],yy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,pt[ci],pt[ci],s3[ci]) \
                +gamma3*ff(xx[ci],yy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,pt[ci],pt[ci],s3[ci]) \
                +gamma4*ff(xx[ci],yy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,pt[ci],pt[ci],s3[ci])
                if ProbP>1e-308:
                    LamdaP = 1/ProbP
                    IniIndLikelihoodP[ci] = (np.log(LamdaP)-LamdaP*Cprior[ci])
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0

    for i in range(1):
        for ci in range(len(xx)):
            if Cprior[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0

    IniLikP = np.sum(IniIndLikelihoodP)*(1/(a.ratio*a.bathsampling))

    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(IniLikP + IniLikF - Max + d/2) 
    return login 

def IniLikelihood2(a,parameter):
    vy1,vy2,vy3,vy4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    Max = a.MaxLogLikeP
    r = a.rr    
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19]
    Dx1 = r[8]
    Dx2 = r[9]
    Dx3 = r[10]
    Dx4 = r[11]      
    Dy1 = r[12]
    Dy2 = r[13]
    Dy3 = r[14]
    Dy4 = r[15]
    xx = a.xx
    DLx = a.DLxx
    yy = a.yy
    DLy = a.DLyy
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04

    DLcon = a.DLconny
    Cprior = a.Cprior
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    ptt = np.array([a.pt]*len(a.xx))
    pt = np.array([a.pt]*len(a.xx))
    st=np.array([a.st]*len(DLx))
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)

    s1 = a.ss1_tcf
    s3 = ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,st[ci],st[ci],s1[ci]) \
                +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,st[ci],st[ci],s1[ci]) \
                +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,st[ci],st[ci],s1[ci]) \
                +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,st[ci],st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    IniLikF = np.sum(IniIndLikelihood)

    IniIndLikelihoodP = np.ones([len(xx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(xx)):
        if Cprior[ci] >0:
            for i in range(1):
                ProbP = gamma1*ff(xx[ci],yy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,pt[ci],pt[ci],s3[ci]) \
                +gamma2*ff(xx[ci],yy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,pt[ci],pt[ci],s3[ci]) \
                +gamma3*ff(xx[ci],yy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,pt[ci],pt[ci],s3[ci]) \
                +gamma4*ff(xx[ci],yy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,pt[ci],pt[ci],s3[ci])
                if ProbP>1e-308:
                    LamdaP = 1/ProbP
                    IniIndLikelihoodP[ci] = (np.log(LamdaP)-LamdaP*Cprior[ci])
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0

    for i in range(1):
        for ci in range(len(xx)):
            if Cprior[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0

    IniLikP = np.sum(IniIndLikelihoodP)*(1/(a.ratio*a.bathsampling))

    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(IniLikP + IniLikF - Max + d/2) 
    return login 

def IniLikelihood3(a,parameter):
    Dx1,Dx2,Dx3,Dx4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    Max = a.MaxLogLikeP
    r = a.rr    
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19]          
    Dy1 = r[12]
    Dy2 = r[13]
    Dy3 = r[14]
    Dy4 = r[15]
    xx = a.xx
    DLx = a.DLxx 
    yy = a.yy
    DLy = a.DLyy 
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04

    DLcon = a.DLconny
    Cprior = a.Cprior
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    ptt = np.array([a.pt]*len(a.xx))
    pt = np.array([a.pt]*len(a.xx))
    st=np.array([a.st]*len(DLx))
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)

    s1 = a.ss1_tcf
    s3 = ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,st[ci],st[ci],s1[ci]) \
                +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,st[ci],st[ci],s1[ci]) \
                +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,st[ci],st[ci],s1[ci]) \
                +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,st[ci],st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    IniLikF = np.sum(IniIndLikelihood)

    IniIndLikelihoodP = np.ones([len(xx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(xx)):
        if Cprior[ci] >0:
            for i in range(1):
                ProbP = gamma1*ff(xx[ci],yy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,pt[ci],pt[ci],s3[ci]) \
                +gamma2*ff(xx[ci],yy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,pt[ci],pt[ci],s3[ci]) \
                +gamma3*ff(xx[ci],yy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,pt[ci],pt[ci],s3[ci]) \
                +gamma4*ff(xx[ci],yy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,pt[ci],pt[ci],s3[ci])
                if ProbP>1e-308:
                    LamdaP = 1/ProbP
                    IniIndLikelihoodP[ci] = (np.log(LamdaP)-LamdaP*Cprior[ci])
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0

    for i in range(1):
        for ci in range(len(xx)):
            if Cprior[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0

    IniLikP = np.sum(IniIndLikelihoodP)*(1/(a.ratio*a.bathsampling))

    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(IniLikP + IniLikF - Max + d/2)
    return login  

def IniLikelihood4(a,parameter):
    Dy1,Dy2,Dy3,Dy4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    params3 = a.fitted_params3
    Dx1,Dx2,Dx3,Dx4 = params3
    Max = a.MaxLogLikeP
    r = a.rr
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19]

    xx = a.xx
    DLx = a.DLxx 
    yy = a.yy
    DLy = a.DLyy 
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04

    DLcon = a.DLconny
    Cprior = a.Cprior
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    ptt = np.array([a.pt]*len(a.xx))
    pt = np.array([a.pt]*len(a.xx))
    st=np.array([a.st]*len(DLx))
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)

    s1 = a.ss1_tcf
    s3 = ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,st[ci],st[ci],s1[ci]) \
                +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,st[ci],st[ci],s1[ci]) \
                +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,st[ci],st[ci],s1[ci]) \
                +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,st[ci],st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    IniLikF = np.sum(IniIndLikelihood)

    IniIndLikelihoodP = np.ones([len(xx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(xx)):
        if Cprior[ci] >0:
            for i in range(1):
                ProbP = gamma1*ff(xx[ci],yy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,pt[ci],pt[ci],s3[ci]) \
                +gamma2*ff(xx[ci],yy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,pt[ci],pt[ci],s3[ci]) \
                +gamma3*ff(xx[ci],yy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,pt[ci],pt[ci],s3[ci]) \
                +gamma4*ff(xx[ci],yy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,pt[ci],pt[ci],s3[ci])
                if ProbP>1e-308:
                    LamdaP = 1/ProbP
                    IniIndLikelihoodP[ci] = (np.log(LamdaP)-LamdaP*Cprior[ci])
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0

    for i in range(1):
        for ci in range(len(xx)):
            if Cprior[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0

    IniLikP = np.sum(IniIndLikelihoodP)*(1/(a.ratio*a.bathsampling))

    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(IniLikP + IniLikF - Max + d/2)
    return login  

def IniLikelihood5(a,parameter):
    ro1,ro2,ro3,ro4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    params3 = a.fitted_params3
    Dx1,Dx2,Dx3,Dx4 = params3
    params4 = a.fitted_params4
    Dy1,Dy2,Dy3,Dy4 = params4
    Max = a.MaxLogLikeP
    r = a.rr
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]

    xx = a.xx
    DLx = a.DLxx 
    yy = a.yy
    DLy = a.DLyy 
    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04

    DLcon = a.DLconny
    Cprior = a.Cprior
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    ptt = np.array([a.pt]*len(a.xx))
    pt = np.array([a.pt]*len(a.xx))
    st=np.array([a.st]*len(DLx))
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)

    s1 = a.ss1_tcf
    s3 = ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,st[ci],st[ci],s1[ci]) \
                +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,st[ci],st[ci],s1[ci]) \
                +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,st[ci],st[ci],s1[ci]) \
                +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,st[ci],st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    IniLikF = np.sum(IniIndLikelihood)

    IniIndLikelihoodP = np.ones([len(xx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(xx)):
        if Cprior[ci] >0:
            for i in range(1):
                ProbP = gamma1*ff(xx[ci],yy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,pt[ci],pt[ci],s3[ci]) \
                +gamma2*ff(xx[ci],yy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,pt[ci],pt[ci],s3[ci]) \
                +gamma3*ff(xx[ci],yy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,pt[ci],pt[ci],s3[ci]) \
                +gamma4*ff(xx[ci],yy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,pt[ci],pt[ci],s3[ci])
                if ProbP>1e-308:
                    LamdaP = 1/ProbP
                    IniIndLikelihoodP[ci] = (np.log(LamdaP)-LamdaP*Cprior[ci])
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0

    for i in range(1):
        for ci in range(len(xx)):
            if Cprior[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0

    IniLikP = np.sum(IniIndLikelihoodP)*(1/(a.ratio*a.bathsampling))

    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(IniLikP + IniLikF - Max + d/2)
    return login 

            
#_______________________________regular likelihoods_________________________________________________

def Likelihood(a,N):
    global parameter
    parameter=sampler(N)
    IniLikelihood=np.array(multicore1(a,parameter))

    IniIndLikelihood = np.transpose(IniLikelihood)
    DLcon = a.DLconny
    DLx = a.DLxx 
    DLy = a.DLyy
    CompLikelihood =np.ones(N)
    Likelihood = np.zeros(N)
    Likelihoodi = np.zeros(N)
    
    for i in range(N):                                                                                                                                                                                                                                  
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci,i] == 0:
                    CompLikelihood[i] = 0
                                                                                                                                                                                                                                
    MaxLogLike=-22
    for i in range(N):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if CompLikelihood[i]==1:
                    Likelihood[i] =  Likelihood[i] + IniIndLikelihood[ci,i]
        if CompLikelihood[i]==1:
            if MaxLogLike ==-22:
                MaxLogLike=Likelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,Likelihood[i]])
    print MaxLogLike

    for i in range(N):
        if CompLikelihood[i]==1:
            Likelihood[i] = Likelihood[i] - MaxLogLike + 7
            Likelihoodi[i] = np.exp(Likelihood[i])

    return Likelihoodi, MaxLogLike, Likelihood

def IniLikelihoodP(a,parameter):
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    
    DLx = a.xx 
    DLy = a.yy
    DLcon = a.Cprior 

    x01=a.xx01
    y01=a.yy01
    x02=a.xx02
    y02=a.yy02
    x03=a.xx03
    y03=a.yy03
    x04=a.xx04
    y04=a.yy04
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04
    st = np.array([a.st]*len(a.xx))
    dt = st 

    ptt = np.array([a.st]*len(a.xx))
    
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    
    s1 = ss3


    IniIndLikelihoodP = np.ones([len(DLx)])
    LamdaP = 0
    ProbP = 0
    CompLikelihoodP=1

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                ProbP = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x01,y01,Dx01,Dy01,dt[ci],st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x02,y02,Dx02,Dy02,dt[ci],st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x03,y03,Dx03,Dy03,dt[ci],st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x04,y04,Dx04,Dy04,dt[ci],st[ci],ss)

                if ProbP>1e-308:
                    LamdaP = 1/ProbP                 
                    IniIndLikelihoodP[ci] = np.log(LamdaP)-LamdaP*DLcon[ci]
                else:
                    LamdaP = 0
                    IniIndLikelihoodP[ci] = 0               
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihoodP[ci] == 0:
                    CompLikelihoodP = 0


    return IniIndLikelihoodP

def integ(a,loc):
    global parameter
    parameter = a.combinepara[0]
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t 
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    ss2 = a.ss2

    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,dt,t,ss2)

    return ProObsGivenPar

def integ2(a,loc):
    global parameter
    parameter = a.combinepara[0]
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(t) == np.min(t):
            dt = np.array(t)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t 
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    ss2 = a.ss2

    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,dt,t,ss2)

    return ProObsGivenPar

def integfin(a,loc):
    global parameter
    parameter = a.ttt
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    ss2 = a.ss2
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array([a.pt]*len(t))
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t 
    mx1=a.xx01
    my1=a.yy01
    mx2=a.xx02
    my2=a.yy02
    mx3=a.xx03
    my3=a.yy03
    mx4=a.xx04
    my4=a.yy04                
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02        
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04           
    ProObsGivenPar = gamma1*ffcont(x,y,vx1,vy1,Dx1,Dy1,ro1,mx1,my1,Dx01,Dy01,dt,t,ss2) \
    +gamma2*ffcont(x,y,vx2,vy2,Dx2,Dy2,ro2,mx2,my2,Dx02,Dy02,dt,t,ss2) \
    +gamma3*ffcont(x,y,vx3,vy3,Dx3,Dy3,ro3,mx3,my3,Dx03,Dy03,dt,t,ss2) \
    +gamma4*ffcont(x,y,vx4,vy4,Dx4,Dy4,ro4,mx4,my4,Dx04,Dy04,dt,t,ss2)

    return ProObsGivenPar

def LikelihoodNew(a,N):
    global parameter
    parameter=sampler(N)
    IniLikelihoodP=np.array(multicore3(a,parameter))
    IniIndLikelihoodP = np.transpose(IniLikelihoodP)
    ratio = a.ratio 
    bathsampling = a.bathsampling
    Depth = a.Depth
    MaxLogLike = a.MaxLogLike
    LikelihoodField = a.LikeField + MaxLogLike - 7

    DLx = a.xx 
    DLcon = a.Cprior 

    CompLikelihoodP =np.ones(N)
    LikelihoodP = np.zeros(N)
    LikelihoodPi = np.zeros(N)
    LikelihoodFandP = np.zeros(N)
    LikelihoodFandPi = np.zeros(N)
    MaxLikelihoodFandP = np.zeros(N)

    if np.min(Depth) == np.max(Depth):
        for i in range(N):
            LikelihoodP[i] = 1.0
            MaxLogLikeP = MaxLogLike
            MaxLogLikePP = MaxLogLike
            LikelihoodFandP[i] = np.exp(LikelihoodField[i]-MaxLogLikePP+7)
    else:
        for i in range(N):                                                                                                                                                                                                                                  
            for ci in range(len(DLx)):
                if DLcon[ci]>0:
                    if IniIndLikelihoodP[ci,i] == 0:
                        CompLikelihoodP[i] = 0
        MaxLogLikeP=-22
        MaxLogLikePP=-22
        for i in range(N):
            for ci in range(len(DLx)):
                if DLcon[ci]>0:
                    if CompLikelihoodP[i]==1:
                        LikelihoodP[i] =  LikelihoodP[i] + IniIndLikelihoodP[ci,i]
                        

        for i in range(N):
            if CompLikelihoodP[i]==1:
                LikelihoodPi[i] = (1/(a.ratio*a.bathsampling))*LikelihoodP[i]
                MaxLikelihoodFandP[i] = LikelihoodPi[i] + LikelihoodField[i] 
            if CompLikelihoodP[i]==1:
                if MaxLogLikePP ==-22:
                    MaxLogLikePP=MaxLikelihoodFandP[i]
                else:
                    MaxLogLikePP=np.max([MaxLogLikePP,MaxLikelihoodFandP[i]])
        print MaxLogLikePP
        for i in range(N):
            if CompLikelihoodP[i]==1:
                LikelihoodFandP[i] = MaxLikelihoodFandP[i] - MaxLogLikePP +7 
                LikelihoodFandP[i] = np.exp(LikelihoodFandP[i])
                

    return MaxLogLikeP, LikelihoodFandP, MaxLogLikePP

def newinteg(a,loc):
    global parameter
    parameter = a.newr
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.t
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    ss2 = a.ss2

    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,dt,t,ss2) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,dt,t,ss2)

    return ProObsGivenPar

def pminteg1(a,loc):
    parameter = a.pm1
    vx1,vy1,Dx1,Dy1,ro1,gamma1 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    ss1 = a.ss2
    x0 = a.xx01
    y0 = a.yy01
    Dx0 = a.ssigmax01
    Dy0 = a.ssigmay01
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t
        
    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss1)
    return ProObsGivenPar

def pminteg2(a,loc):
    parameter = a.pm2
    vx1,vy1,Dx1,Dy1,ro1,gamma1 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    ss1 = a.ss2
    x0 = a.xx02
    y0 = a.yy02
    Dx0 = a.ssigmax02
    Dy0 = a.ssigmay02
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t
        
    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss1)
    return ProObsGivenPar

def pminteg3(a,loc):
    parameter = a.pm3
    vx1,vy1,Dx1,Dy1,ro1,gamma1 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    ss1 = a.ss2
    x0 = a.xx02
    y0 = a.yy02
    Dx0 = a.ssigmax03
    Dy0 = a.ssigmay03
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t
        
    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss1)
    return ProObsGivenPar

def pminteg4(a,loc):
    parameter = a.pm4
    vx1,vy1,Dx1,Dy1,ro1,gamma1 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.tt
    ss1 = a.ss2
    x0 = a.xx04
    y0 = a.yy04
    Dx0 = a.ssigmax04
    Dy0 = a.ssigmay04
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array(a.pt)
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t
        
    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,dt,t,ss1)
    return ProObsGivenPar

def integcf(a,loc):
    global parameter
    parameter = a.ppar
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0
    [x,y]=loc
    t=a.tt
    ss2 = a.ss2
    if a.ourinformation['SpillPlace'] == 'River':
        if np.max(a.st) == np.min(a.st):
            dt = np.array([a.pt]*len(t))
        else:
            dt = t 
    if a.ourinformation['SpillPlace'] == 'Ocean':
        dt = t 
    mx1=a.xx01
    my1=a.yy01
    mx2=a.xx02
    my2=a.yy02
    mx3=a.xx03
    my3=a.yy03
    mx4=a.xx04
    my4=a.yy04                
    Dx01=a.ssigmax01
    Dy01=a.ssigmay01
    Dx02=a.ssigmax02
    Dy02=a.ssigmay02        
    Dx03=a.ssigmax03
    Dy03=a.ssigmay03
    Dx04=a.ssigmax04
    Dy04=a.ssigmay04           
    ProObsGivenPar = gamma1*ffcont(x,y,vx1,vy1,Dx1,Dy1,ro1,mx1,my1,Dx01,Dy01,dt,t,ss2) \
    +gamma2*ffcont(x,y,vx2,vy2,Dx2,Dy2,ro2,mx2,my2,Dx02,Dy02,dt,t,ss2) \
    +gamma3*ffcont(x,y,vx3,vy3,Dx3,Dy3,ro3,mx3,my3,Dx03,Dy03,dt,t,ss2) \
    +gamma4*ffcont(x,y,vx4,vy4,Dx4,Dy4,ro4,mx4,my4,Dx04,Dy04,dt,t,ss2)

    return ProObsGivenPar


def multicore1(a,parameter):
        pool = mp.Pool(15)
        res = pool.map(partial(IniLikelihood,a),parameter)
        return res


def multicore2(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integ,a),loc)
        return res

def multicore21(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integ2,a),loc)
        return res


def multicore3(a,parameter):
        pool = mp.Pool(15)
        res = pool.map(partial(IniLikelihoodP,a),parameter)
        return res


def multicore4(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(newinteg,a),loc)
        return res

def multicore5(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integcf,a),loc)
        return res

def multicore6(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(pminteg1,a),loc)
        return res

def multicore7(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(pminteg2,a),loc)
        return res

def multicore8(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(pminteg3,a),loc)
        return res

def multicore9(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(pminteg4,a),loc)
        return res
def multicore10(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integfin,a),loc)
        return res

def multicore11(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integfincont,a),loc)
        return res

def extract_key(v):
        return v[0]

class Preliminars: # SOSim
    def __init__(self): 
        self.w = 4
        self.u = self.w + 1

class soscore(Preliminars):
    def __init__(self,datalist):
        Preliminars.__init__(self)
        #Input data

        lat = []
        lat.append(datalist['lat'])
        lon = []
        lon.append(datalist['lon'])
        SpillT = []
        SpillT.append(datalist['starttime'])
        SpillT.append(datalist['endtime'])
        
        PredictT = []
        PredictT.append(datalist['PredictTime'])
        OilType = []
        OilType.append(datalist['OilType'])
        Scale = []
        Scale.append(float(datalist['lonscale']))
        Scale.append(float(datalist['latscale']))
        Node = []
        Node.append(float(datalist['xNode']))
        Node.append(float(datalist['yNode']))
        SpillPlace = datalist['SpillPlace']
        BathyUpload = str(ourinformation['SunkenUpload'])
        UploadType = str(datalist['SunkenUpload'])

        lat = np.array(lat)[~np.isnan(lat)]
        lon = np.array(lon)[~np.isnan(lon)]
        SpillT = np.array(SpillT)[~pd.isnull(SpillT)]
        PredictT = np.array(PredictT)[~pd.isnull(PredictT)]
        Scale = np.array(Scale)[~np.isnan(Scale)]
        Node = np.array(Node)[~np.isnan(Node)]
        OilType = np.array(OilType)[~np.isnan(OilType)]

        if OilType == 1:
            retardation = 5.6
        if OilType == 2:
            retardation = 4.2
        if OilType == 3:
            retardation = 2.8
        if OilType == 4:
            retardation = 1.4
        if OilType == 5:
            retardation = 0.0

        sigmax0 = 0.050
        sigmay0 = 0.050
        #define spill point
        coord0 = utm.from_latlon(lat, lon)
        x0 = coord0[0]/1000.0
        y0 = coord0[1]/1000.0
        
        duration = [CalTime(SpillT[0],SpillT[1])]
        smallspill = np.linspace(0,duration[0],duration[0]*30)
        dura = np.pad(duration,(0,1),'constant')
        t = [CalTime(SpillT[0],PredictT[vld]) for vld in range(len(PredictT))]

        ss2 = []

        for i in range(len(t)):
            after = CalTime(PredictT[i],SpillT[1]) 
            before = CalTime(SpillT[0],PredictT[i]) - retardation 
            if dura[0] != dura[1] and after > 0:
                K = ceil(before)
                s2 = np.linspace(0.0,before,K)
                ss2.append(s2)
            elif SpillT[0] < SpillT[1]:
                K = ceil(dura[0])+1
                s2 = np.linspace(dura[1],dura[0]-retardation,K)
                ss2.append(s2)
            else:
                K = 1
                s2 = np.zeros(K)
                ss2.append(s2)

        ss2 = np.array(ss2)

        self.x0 = x0
        self.y0 = y0
        #self.x01 = self.x02 = self.x03 = self.x04 = x0
        #self.y01 = self.y02 = self.y03 = self.y04 = y0
        self.xxx01 = self.xxx02 = self.xxx03 = self.xxx04 = x0
        self.yyy01 = self.yyy02 = self.yyy03 = self.yyy04 = y0
        self.ssigmax01 = self.ssigmax02 = self.ssigmax03 = self.ssigmax04 = sigmax0
        self.ssigmay01 = self.ssigmay02 = self.ssigmay03 = self.ssigmay04 = sigmay0
        self.Dx01 = self.Dx02 = self.Dx03 = self.Dx04 = sigmax0
        self.Dy01 = self.Dy02 = self.Dy03 = self.Dy04 = sigmay0 
        self.SpillT = SpillT
        self.PredictT = PredictT
        self.Scale = Scale
        self.xNode = Node[0]
        self.yNode = Node[1]
        self.OilType = OilType
        self.SpillPlace = SpillPlace
        self.UploadType = UploadType
        self.BathyUpload = BathyUpload
        self.Dx0 = sigmax0
        self.Dy0 = sigmay0
        self.x00 = lat
        self.y00 = lon
        self.t = t
        self.duration = duration
        self.dura = dura
        self.scale = Scale
        self.lat0 = lat 
        self.lon0 = lon 
        self.ss2 = ss2
        self.retardation = retardation 
        self.smallspill = smallspill

        #Load campaign data
    def UploadCampaign(self,CampaignFileName):
        DLx = []
        DLy = []
        DLcon = []
        st = [] 
        dx = []
        dy = []
        dcon = []
        ST = []
        DLcl = []
        f=[]
        lati=[]
        longi=[]
        lat = []
        lon = []
        con = []
        s = []

        print "CampaignFileName",CampaignFileName

        for i in range(len(CampaignFileName)):
            if i == 0:
                campdata = pd.read_csv(CampaignFileName[i])
            else:
                campdata = pd.concat([campdata , pd.read_csv(CampaignFileName[i])],axis = 0,ignore_index=True)


        SampleT = campdata['SampleTime']
        SampleT = SampleT[~pd.isnull(SampleT)]
        DLlat = np.array(campdata["latitude"])
        DLlon = np.array(campdata["longitude"])
        DLc = np.array(campdata["total_con"])
        lat.extend(DLlat)
        lon.extend(DLlon)
        con.extend(DLc)
        s.extend(SampleT)
        stt=np.array(s)
       
        for j in range(len(s)):  
            cc = CalTime(self.SpillT[0],s[j])
            st.append(cc)   

        ccord = zip(DLlat,DLlon)
        coordcon = zip(ccord,DLc,st)
        data = sorted(coordcon,key=extract_key)
        result = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(data, extract_key)]

        tt=[]
        for i in range(len(result)):
            conres=[list(p) for m, p in itertools.groupby(result[i][1],lambda x:x[1])]
            for j in conres:
                jarray=np.array(j)
                con = np.mean(jarray[:,0])
                DLcl.append(con)            
                h = result[i][0]
                f.append(h)
                tt.append(j[0][1])

        tt = np.array(tt)

        for i in range(len(f)):
            l=f[i][0]
            t=f[i][1]
            lati.append(l)
            longi.append(t)

        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(DLlat,DLlon)])
        DLx.append(np.array(map(float,camdatalist[:,0]))/1000)

        DLy.append(np.array(map(float,camdatalist[:,1]))/1000)
        DLx = DLx[0]
        DLy = DLy[0]
#            
             
        for s in DLc:
            if s == 0.0:
                conValue = (0.0000001)
            else:
                conValue = s
            DLcon.append(conValue)
            

        DLcon = DLcon/np.max(DLcon)
        DLcon = np.array(DLcon)

        ss1 = []

        st = np.array(st) - self.retardation
        
        uni = np.unique(st) 

        DLxtv = np.array([DLx[np.where(st==i)] for i in np.unique(st)])    
        DLytv = np.array([DLy[np.where(st==i)] for i in np.unique(st)]) 
        newst = np.array([st[np.where(st==i)] for i in np.unique(st)])        
        DLcontv = np.array([DLcon[np.where(st==i)] for i in np.unique(st)])


        for i in range(len(uni)):
            if i == 0:
                after = CalTime(SampleT[i],self.SpillT[1])
                before = CalTime(self.SpillT[0],SampleT[i])-self.retardation

                if self.dura[0] != self.dura[1] and after > 0:
                    K = ceil(before)+1
                    s1 = np.linspace(0.0,before,K)
                    ss1.append([s1]*len(DLxtv[i]))
                elif self.SpillT[0] < self.SpillT[1]:
                    K = ceil(self.dura[0])+1
                    s1 = np.linspace(self.dura[1],self.dura[0]-self.retardation,K)
                    ss1.append([s1]*len(DLxtv[i]))
                else:
                    K = 1 
                    s1 = np.zeros(K)
                    ss1.append([s1]*len(DLxtv[i]))
            else:
                after = CalTime(SampleT[i],self.SpillT[1])

                if self.dura[0] != self.dura[1] and after > 0:
                    K = ceil(uni[i]-uni[i-1])+1
                    s1 = np.linspace(0.0,uni[i]-uni[i-1],K)
                    ss1.append([s1]*len(DLxtv[i]))
                elif self.SpillT[0] < self.SpillT[1]:
                    K = ceil(self.dura[0])+1
                    s1 = np.linspace(self.dura[1],self.dura[0]-self.retardation,K)
                    ss1.append([s1]*len(DLxtv[i]))
                else:
                    K = 1 
                    s1 = np.zeros(K)
                    ss1.append([s1]*len(DLxtv[i]))



        self.DLx = DLx 
        self.DLy = DLy 
        self.DLcon = DLcon
        self.DLxtv = DLxtv 
        self.DLytv = DLytv 
        self.DLcontv = DLcontv 
        self.newst = newst
        self.uniST = uni 
        self.st = st
        self.lati = lati 
        self.longi = longi 
        self.ss1 = ss1
        self.DLlat = DLlat 
        self.DLlon = DLlon

    #function to be used if the user clicks 'Bathymetry Upload'
    def UploadBathymetry(self,BathymetryFile):
        bat = []
        batt = []
        Dep = []
        xd = []
        yd = []
        BathymetryFileName = []
        BathymetryFileName.append(BathymetryFile)

        for i in range(len(BathymetryFileName)):
            bathdata= pd.read_csv(BathymetryFileName[i])
            
            # if 'UTM coord' is clicked
            if ourinformation['SunkenUpload'] == 'UTM coord':
                print "UTM coord" 
                if ourinformation['SpillPlace'] == 'River':
                    x = np.array(bathdata["Easting"])
                    x = x/1000.
                    y = np.array(bathdata["Northing"])
                    y = y/1000.
                    d = np.array(bathdata["Depth"])
                    xdd = x 
                    ydd = y 
                if ourinformation['SpillPlace'] == 'Ocean':
                    utmspill = utm.from_latlon(self.lat0,self.lon0)
                    x = np.array(bathdata["Easting"])
                    y = np.array(bathdata["Northing"])
                    d = np.array(bathdata["Depth"])
                    camdatalist = np.array([utm.to_latlon(i,j,utmspill[2],utmspill[3]) for i,j in zip(x,y)])
                    for i in range(len(camdatalist)):
                        xd.append(camdatalist[i][0])
                        yd.append(camdatalist[i][1])

                    xdd = np.array(xd)
                    ydd = np.array(yd)


            if ourinformation['SunkenUpload'] == 'Decimal degrees':
                print "Decimal degrees"
                xdd = np.array(bathdata["latitude"])
                ydd = np.array(bathdata["longitude"])
                d = np.array(bathdata["depth"])

        if ourinformation['SpillPlace'] == 'River':
            xlow = np.min(self.newx)
            xhigh = np.max(self.newx)
            ylow = np.min(self.newy)
            yhigh = np.max(self.newy)

        if ourinformation['SpillPlace'] == 'Ocean':
            xlow = self.lat0 - self.scale[0]
            xhigh = self.lat0 + self.scale[0]
            ylow = self.lon0 - self.scale[1]
            yhigh = self.lon0 + self.scale[1]

        xl = zip(xdd,ydd,d)
        for i in range(len(xl)):
            if xlow <= xl[i][0] <= xhigh and ylow <= xl[i][1] <= yhigh:
                bat.append(xl[i][0])
                batt.append(xl[i][1])
                Dep.append(xl[i][2])
        bat = np.array(bat)
        batt = np.array(batt)
        Dep = np.array(Dep)
        self.bat = bat 
        self.batt = batt 
        self.Dep = Dep 

    def retardationDueOilType(self):
        if self.OilType == 1.0:
            retardation = 5.6
        if self.OilType == 2.0:
            retardation = 4.2
        if self.OilType == 3.0:
            retardation = 2.8
        if self.OilType == 4.0:
            retardation = 1.4
        if self.OilType == 5.0:
            retardation = 0.0
        self.retardation = retardation
        self.st = self.st - retardation
        self.pt = self.st[0]

        K = 1
        self.ss3 = np.zeros(K)
        B = [max(self.DLcon)]
 
        hiIndex = B.index(max(B))
        latestST = max(self.st)
        self.t = np.array(self.t) - retardation 


    def x0y0DueSinkingRetardation(self):
        B = np.max(self.DLcon)
        C = np.argmax(self.DLcon)     
    
        x0news = self.DLx[C]
        y0news = self.DLy[C]
        
        x0new = x0news
        y0new = y0news

        x0 = self.x0
        y0 = self.y0
        xxx01 = self.xxx01
        yyy01 = self.yyy01
        xxx02 = self.xxx02
        yyy02 = self.yyy02
        xxx03 = self.xxx03
        yyy03 = self.yyy03
        xxx04 = self.xxx04
        yyy04 = self.yyy04
        oilType = self.OilType

        distX = np.array(x0new - x0)
        distY = np.array(y0new - y0)
        if oilType == 1.0:
            sunkx0 = (x0 + (5.6*(np.array(distX)/8.0)))
            sunky0 = (y0 + (5.6*(np.array(distY)/8.0)))
            sunkxx01 = (xxx01 + (5.6*(np.array(distX)/8.0)))
            sunkyy01 = (yyy01 + (5.6*(np.array(distY)/8.0)))
            sunkxx02 = (xxx02 + (5.6*(np.array(distX)/8.0)))
            sunkyy02 = (yyy02 + (5.6*(np.array(distY)/8.0)))
            sunkxx03 = (xxx03 + (5.6*(np.array(distX)/8.0)))
            sunkyy03 = (yyy03 + (5.6*(np.array(distY)/8.0)))
            sunkxx04 = (xxx04 + (5.6*(np.array(distX)/8.0)))
            sunkyy04 = (yyy04 + (5.6*(np.array(distY)/8.0)))
        if oilType == 2.0:
            sunkx0 = (x0 + (4.2*(np.array(distX)/8.0)))
            sunky0 = (y0 + (4.2*(np.array(distY)/8.0)))
            sunkxx01 = (xxx01 + (4.2*(np.array(distX)/8.0)))
            sunkyy01 = (yyy01 + (4.2*(np.array(distY)/8.0)))
            sunkxx02 = (xxx02 + (4.2*(np.array(distX)/8.0)))
            sunkyy02 = (yyy02 + (4.2*(np.array(distY)/8.0)))
            sunkxx03 = (xxx03 + (4.2*(np.array(distX)/8.0)))
            sunkyy03 = (yyy03 + (4.2*(np.array(distY)/8.0)))
            sunkxx04 = (xxx04 + (4.2*(np.array(distX)/8.0)))
            sunkyy04 = (yyy04 + (4.2*(np.array(distY)/8.0)))
        if oilType == 3.0:
            sunkx0 = (x0 + (2.8*(np.array(distX)/8.0)))
            sunky0 = (y0 + (2.8*(np.array(distY)/8.0)))
            sunkxx01 = (xxx01 + (2.8*(np.array(distX)/8.0)))
            sunkyy01 = (yyy01 + (2.8*(np.array(distY)/8.0)))
            sunkxx02 = (xxx02 + (2.8*(np.array(distX)/8.0)))
            sunkyy02 = (yyy02 + (2.8*(np.array(distY)/8.0)))
            sunkxx03 = (xxx03 + (2.8*(np.array(distX)/8.0)))
            sunkyy03 = (yyy03 + (2.8*(np.array(distY)/8.0)))
            sunkxx04 = (xxx04 + (2.8*(np.array(distX)/8.0)))
            sunkyy04 = (yyy04 + (2.8*(np.array(distY)/8.0)))
        if oilType == 4.0:
            sunkx0 = (x0 + (1.4*(np.array(distX)/8.0)))
            sunky0 = (y0 + (1.4*(np.array(distY)/8.0)))
            sunkxx01 = (xxx01 + (1.4*(np.array(distX)/8.0)))
            sunkyy01 = (yyy01 + (1.4*(np.array(distY)/8.0)))
            sunkxx02 = (xxx02 + (1.4*(np.array(distX)/8.0)))
            sunkyy02 = (yyy02 + (1.4*(np.array(distY)/8.0)))
            sunkxx03 = (xxx03 + (1.4*(np.array(distX)/8.0)))
            sunkyy03 = (yyy03 + (1.4*(np.array(distY)/8.0)))
            sunkxx04 = (xxx04 + (1.4*(np.array(distX)/8.0)))
            sunkyy04 = (yyy04 + (1.4*(np.array(distY)/8.0)))
        if oilType == 5.0:
            sunkx0 = (x0 + (0.0*(np.array(distX)/8.0)))
            sunky0 = (y0 + (0.0*(np.array(distY)/8.0)))
            sunkxx01 = (xxx01 + (0.0*(np.array(distX)/8.0)))
            sunkyy01 = (yyy01 + (0.0*(np.array(distY)/8.0)))
            sunkxx02 = (xxx02 + (0.0*(np.array(distX)/8.0)))
            sunkyy02 = (yyy02 + (0.0*(np.array(distY)/8.0)))
            sunkxx03 = (xxx03 + (0.0*(np.array(distX)/8.0)))
            sunkyy03 = (yyy03 + (0.0*(np.array(distY)/8.0)))
            sunkxx04 = (xxx04 + (0.0*(np.array(distX)/8.0)))
            sunkyy04 = (yyy04 + (0.0*(np.array(distY)/8.0)))
        self.sunkx0 = sunkx0
        self.sunky0 = sunky0
        self.xx01 = sunkxx01
        self.xx02 = sunkxx02
        self.xx03 = sunkxx03
        self.xx04 = sunkxx04
        self.yy01 = sunkyy01
        self.yy02 = sunkyy02
        self.yy03 = sunkyy03
        self.yy04 = sunkyy04
        self.x01 = self.x02 = self.x03 = self.x04 = sunkx0
        self.y01 = self.y02 = self.y03 = self.y04 = sunky0


def sunken_main(myinformation,progressBar):

    print myinformation

    global ourinformation

    ourinformation = myinformation
    a = soscore(ourinformation)
    a.ourinformation = ourinformation
    progressBar.setValue(15)


    a.UploadCampaign(myinformation['CampaignButton'])
    
    a.retardationDueOilType()
    a.x0y0DueSinkingRetardation() 

    SX = a.DLx
    SY = a.DLy
    lat0 = a.lat0[0]
    lon0 = a.lon0[0]
    x0 = a.sunkx0
    y0 = a.sunky0
    SpillPlace = a.SpillPlace
    a.SpillPlace = SpillPlace
    BathyUpload = a.BathyUpload


    DLlat = a.DLlat 
    DLlon = a.DLlon

    XSpill = a.sunkx0
    YSpill = a.sunky0

    if ourinformation['SpillPlace'] == 'River':
        N=500000
        XXa = np.linspace(lat0-a.scale[1],lat0+a.scale[1],2000)
        YYa = np.linspace(lon0-a.scale[0],lon0+a.scale[0],2000)
        db = oceansdb.ETOPO()
        dd = db['topography'].extract(lat=XXa,lon=YYa)
        griddepth = dd['height']
        [yyy,xxx] = np.meshgrid(YYa,XXa)
        loc = []
        for i in range(len(griddepth)):
            loc.append(zip(xxx[i],yyy[i],griddepth[i]))
        newxx = []
        newyy = []
        newd = []
        for i in range(len(loc)):
            for j in range(len(loc[i])):
                newxx.append(loc[i][j][0])
                newyy.append(loc[i][j][1])
                newd.append(loc[i][j][2])
        newxx = np.array(newxx)
        newyy = np.array(newyy)
        newd = np.array(newd)
        newloc = zip(newxx,newyy,newd)
        newlocx = []
        newlocy = []
        pdepth = []
        for i in range(len(newloc)):
            if newloc[i][2] < 0:
                newlocx.append(newloc[i][0])
                newlocy.append(newloc[i][1])
                pdepth.append(newloc[i][2])
        newlocx = np.array(newlocx)
        newlocy = np.array(newlocy)
        pdepth = np.array(pdepth) 

        newx = newlocx 
        newy = newlocy
        coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(newx,newy)])
        xa = np.array(map(float,coord[:,0]))/1000
        ya = np.array(map(float,coord[:,1]))/1000
        x = xa 
        y = ya

        xaa = newx[0::len(newx)//int(a.xNode)]
        yaa = newy[0::len(newy)//int(a.yNode)]

        a.xa = xa
        a.ya = ya 
        a.newx = newx 
        a.newy = newy 

    if ourinformation['SpillPlace'] == 'Ocean':
        N=1000000
        XXa = np.linspace(lat0-a.scale[1],lat0+a.scale[1],a.xNode+1)
        YYa = np.linspace(lon0-a.scale[0],lon0+a.scale[0],a.yNode+1)
        coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(XXa,YYa)])
        xa = np.array(map(float,coord[:,0]))/1000
        ya = np.array(map(float,coord[:,1]))/1000
        [x,y] = np.meshgrid(xa,ya)
        x = np.concatenate(x)
        y = np.concatenate(y)
        newx = XXa 
        newy = YYa
    
    Xa = np.linspace(lat0-a.scale[1],lat0+a.scale[1],a.xNode+1)
    Ya = np.linspace(lon0-a.scale[0],lon0+a.scale[0],a.yNode+1)

    ratio = float(ourinformation["Ratio"])
    print 'ratio',ratio
    a.ratio = ratio 
    

    X=SX
    Y=SY
    t = a.t
    print 't',t
    SP = a.st
    pt = a.pt
    ss2 = a.ss2
    ss3 = a.ss3
    Dx0 = 0.05
    Dy0 = 0.05

    a.x0=a.sunkx0
    a.y0=a.sunky0

    
    a.stind = np.transpose(a.uniST)  
    a.stind = np.insert(a.stind,0,-1)    
    a.stindnew = a.stind[1:5]
    priorxx = []
    prioryy = []
    priorccon = []
    priorYY = []  
    combinepara = []   
    combineparanew = []  
    MaxLog = []
    MaxLogP = [] 
    prob = []
    stpredict = []
    for k in range(len(a.DLxtv)):
        a.DLx_t = a.DLxtv[k]
        a.DLy_t = a.DLytv[k]
        a.DLcon_t = a.DLcontv[k]
        a.ss1_t = a.ss1[k]
        if k == 0: 
            a.st = a.newst[k][0]
            stpredict.append(a.st)
        else: 
            a.st = a.newst[k][0]-a.newst[k-1][0]
            stpredict.append(a.st)
        a.STT = [0]
        t=a.stind[k+1]
        DDLx = a.DLx_t
        DDLy = a.DLy_t
        DDLcon = a.DLcon_t
        DDss1 = a.ss1_t 

        a.xx01 = [a.xx01][0]
        
        a.yy01 = [a.yy01][0]
        a.xx02 = [a.xx02][0]
        a.yy02 = [a.yy02][0]
        a.xx03 = [a.xx03][0]
        a.yy03 = [a.yy03][0]
        a.xx04 = [a.xx04][0]
        a.yy04 = [a.yy04][0]

        print 'a.xx01',a.xx01, a.xx02, a.xx03, a.xx04
        print 'a.yy01',a.yy01, a.yy02, a.yy03, a.yy04 
                                      
        a.ssigmax01=[a.ssigmax01][0]                
        a.ssigmay01=[a.ssigmay01][0]
        a.ssigmax02=[a.ssigmax02][0]
        a.ssigmay02=[a.ssigmay02][0]
        a.ssigmax03=[a.ssigmax03][0]
        a.ssigmay03=[a.ssigmay03][0]
        a.ssigmax04=[a.ssigmax04][0]
        a.ssigmay04=[a.ssigmay04][0]

        print 'a.ssigmax',a.ssigmax01, a.ssigmax02, a.ssigmax03, a.ssigmax04
        print 'a.ssigmay',a.ssigmay01, a.ssigmay02, a.ssigmay03, a.ssigmay04

        parameter = sampler(N)
        
        a.DLxx = DDLx 
        a.DLyy = DDLy 
        a.DLconny = DDLcon
        a.ss11 = DDss1

        print 'sample time', a.st

        if ourinformation['Run'] == 'Run':
            prob, MaxLogg, LikeField = Likelihood(a,N)
            a.LikeField = LikeField
            a.prob = prob
            a.MaxLogLike = MaxLogg
            global parameter
            parameter = sampler(N)
            print 'parameter[0]',parameter[0]
            Likemle = np.argmax(prob)
            a.r = parameter[Likemle]
            print 'MLE parameter', a.r
            combinepara.append(a.r)
            a.combinepara = combinepara
            MaxLog.append(MaxLogg)


        if ourinformation['Run'] == 'Recalc':
            if len(a.DLxtv) == 1:
                paramsload = [np.loadtxt('fieldparameters.txt',delimiter=',')]
                MaxLog = [np.loadtxt('MaxLogLike.txt',delimiter=',')]
                a.r = [paramsload[k]]
                a.combinepara = [paramsload[k]]
                combinepara = [paramsload[k]]
            else:
                paramsload = np.loadtxt('fieldparameters.txt',delimiter=',')
                MaxLog = np.loadtxt('MaxLogLike.txt',delimiter=',')
                a.r = paramsload[k]
                a.combinepara = [paramsload[k]]
                combinepara = paramsload
            
            MaxLog = MaxLog[k]


        t=a.t
        res = []
        for u in range(len(t)):
            a.tt = a.newst[0][0]
            a.ss2 = a.ss1[0][0]
            resa = []
            loc = zip(x,y)
            resa=np.array(multicore2(a,loc))
            resa=np.transpose(resa)
            sum = 0
            res.append(resa)
        sfield = np.array(res)
        #print 'field2', list(sfield)
        sfield = sfield/np.max(sfield)
        

        pm1 = combinepara[k][0],combinepara[k][4],combinepara[k][8],combinepara[k][12],combinepara[k][16],combinepara[k][20]
        pm2 = combinepara[k][1],combinepara[k][5],combinepara[k][9],combinepara[k][13],combinepara[k][17],combinepara[k][21]
        pm3 = combinepara[k][2],combinepara[k][6],combinepara[k][10],combinepara[k][14],combinepara[k][18],combinepara[k][22]
        pm4 = combinepara[k][3],combinepara[k][7],combinepara[k][11],combinepara[k][15],combinepara[k][19],combinepara[k][23]
        a.pm1 = pm1
        a.pm2 = pm2 
        a.pm3 = pm3
        a.pm4 = pm4

        Conmax = np.argmax(DDLcon)
        m = np.max(DDLcon)
        zxmax = [i for i,j in enumerate(DDLcon) if j == m]
        xxmax = DDLx[zxmax]
        yymax = DDLy[zxmax]
        SPPP = a.st
        distx = abs(x0-xxmax)
        disty = abs(y0-yymax) 
        dist = np.sqrt((distx**2)+(disty**2))
        closedist = np.argmin(dist)
        xCmax = DDLx[Conmax]
        yCmax = DDLy[Conmax]
        print 'x0,y0', x0,y0
        SPP = a.st



        a.x=x
        a.y=y
        t=a.t
        respm1 = []
        for u in range(len(t)):
            a.tt = a.st 
            a.ss2 = DDss1[0]
            loc = [(xCmax,yCmax)]
            resa=np.array(multicore6(a,loc))
            resa=np.transpose(resa)
            sum = 0
            respm1.append(resa)
        spm1 = np.array(respm1)

        respm2 = []
        for u in range(len(t)):
            a.tt = a.st 
            a.ss2 = DDss1[0]
            loc = [(xCmax,yCmax)]
            resa=np.array(multicore7(a,loc))
            resa=np.transpose(resa)
            sum = 0
            respm2.append(resa)
        spm2 = np.array(respm2)

        respm3 = []
        for u in range(len(t)):
            a.tt = a.st 
            a.ss2 = DDss1[0]
            loc = [(xCmax,yCmax)] 
            resa=np.array(multicore8(a,loc))
            resa=np.transpose(resa)
            sum = 0
            respm3.append(resa)
        spm3 = np.array(respm3)

        respm4 = []
        for u in range(len(t)):
            a.tt = a.st 
            a.ss2 = DDss1[0]
            loc = [(xCmax,yCmax)]
            resa=np.array(multicore9(a,loc))
            resa=np.transpose(resa)
            sum = 0
            respm4.append(resa)
        spm4 = np.array(respm4)


        maxm1 = np.max(spm1)
        maxm2 = np.max(spm2)
        maxm3 = np.max(spm3)
        maxm4 = np.max(spm4)

        allmaxm = maxm1,maxm2,maxm3,maxm4
        gmax = np.argmax(allmaxm)
        
        rr = np.array(combinepara[k])

        vxmle = rr[0:4]
        vxm = vxmle[gmax]
        vymle = rr[4:8]
        vym = vymle[gmax]
        Dxmle = rr[8:12]
        Dxm = Dxmle[gmax] 
        Dymle = rr[12:16] 
        Dym = Dymle[gmax]
        romle = rr[16:20]
        rom = romle[gmax]

        allxx0 = a.xx01,a.xx02,a.xx03,a.xx04
        allyy0 = a.yy01,a.yy02,a.yy03,a.yy04
        allsigxx0 = a.ssigmax01,a.ssigmax02,a.ssigmax03,a.ssigmax04
        allsigyy0 = a.ssigmay01,a.ssigmay02,a.ssigmay03,a.ssigmay04

        xx0m = allxx0[gmax]
        yy0m = allyy0[gmax]
        sigx0m = allsigxx0[gmax]
        sigy0m = allsigyy0[gmax]



        kx = (xCmax - (xx0m + vxm*SPP))/(sigx0m + np.sqrt(2.0*Dxm*SPP))
        ky = (yCmax - (yy0m + vym*SPP))/(sigy0m + np.sqrt(2.0*Dym*SPP))
        Pmaxnorm = np.max(DDLcon)/(np.exp(((kx**2.0)+(ky**2.0)-(2.0*rom*kx*ky))/(2.0*(1.0-(rom**2.0)))))
        print 'Pmaxnorm',Pmaxnorm
        #This part is if the User clicks the 'No Upload' button.
        
        if ourinformation['SunkenUpload'] == 'No Upload':
            print "No Upload"
            ######################## calculating f/h contours ###################################
            db = oceansdb.ETOPO()
            dfhcont = db['topography'].extract(lat=Xa, lon=Ya)
            dfh = dfhcont['height']
            depfh = np.ndarray.flatten(dfh)

            latrep = np.repeat(Xa,a.xNode+1)
            Latradians = []
            for i in range(len(latrep)):
                Latradians.append(radians(latrep[i]))
            Latradians = np.array(Latradians)

            coriolis = 2*(7.2921*10**-5)*np.sin(Latradians)
            fhcont = coriolis/abs(depfh)

            cs = plt.contour(YYa,XXa,fhcont.reshape(len(XXa),len(YYa)))

            p = []
            v = []
            contlevels = cs.levels
            for i in range(len(cs.collections)):
                p.append(cs.collections[i].get_paths())
                vv = []
                for j in range(len(p[i])):
                    vv.append(p[i][j].vertices)
                
                v.append(np.concatenate(vv))


            xpv = []
            ypv = []
            for i in range(len(v)):
                xpv1 = []
                ypv1 = []
                for j in range(len(v[i])):
                    xpv1.append(float(v[i][j][0]))
                    ypv1.append(float(v[i][j][1]))
                xpv.append(xpv1)
                ypv.append(ypv1)

            lengthcontours = []
            for i in range(len(xpv)):
                lengthcontours.append(len(xpv[i]))

            priorsample = floor((len(a.DLxx)*10)/len(xpv))
            priorsample = int(priorsample)
            print 'priorsample', priorsample


            xbath = []
            ybath = []
            for i in range(len(xpv)):
                if lengthcontours[i] > priorsample:
                    xbath.append(np.random.choice(xpv[i],priorsample))
                    ybath.append(np.random.choice(ypv[i],priorsample))
                else:
                    xbath.append(np.random.choice(xpv[i],lengthcontours[i]))
                    ybath.append(np.random.choice(ypv[i],lengthcontours[i]))

            xbathh = [item for sublist in xbath for item in sublist]
            ybathh = [item for sublist in ybath for item in sublist]
            xbathh = np.array(xbathh)
            ybathh = np.array(ybathh)



            newrat = len(xbathh)/len(a.DLxx)
            bathsampling = floor(newrat)
            a.bathsampling = bathsampling 

            depth = np.zeros(len(xbathh))
            for i in range(len(xbathh)):
                depthh = db['topography'].extract(lat=ybathh[i], lon=xbathh[i])
                depth[i] = depthh['height']

            depth = np.array(depth)

            print 'length prior', len(depth)

            coordp = np.array([utm.from_latlon(i,j) for (i,j) in zip(ybathh,xbathh)])
            xp = np.array(map(float,coordp[:,0]))/1000
            yp = np.array(map(float,coordp[:,1]))/1000

            xx = xp 
            yy = yp  
            Depth = depth

            

         #This part is if the User clicks the 'Bathymetry Upload' button.
        
        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
            #this function 'UploadBathymetry' is used if the 'Bathymetry Upload' button is clicked.  The user clicked 'UTM coord' or 'Decimal degrees' and one of these buttons is connected to this function.
            
            print "Bathymetry Upload"
            a.UploadBathymetry(ourinformation['HydroButton'])

            bat = a.bat 
            batt = a.batt 
            Dep = a.Dep


            if ourinformation['SpillPlace'] == 'Ocean':
                if len(bat) > 2000:
                    bat = bat[0::len(bat)//2000]
                    batt = batt[0::len(batt)//2000]
                    Dep = Dep[0::len(Dep)//2000]
                else:
                    bat = bat 
                    batt = batt 
                    Dep = Dep 
                Latradians = []
                for i in range(len(bat)):
                    Latradians.append(radians(bat[i]))
                Latradians = np.array(Latradians)

                coriolis = 2*(7.2921*10**-5)*np.sin(Latradians)
                fhcont = coriolis/abs(Dep)

                cs = plt.tricontour(batt,bat,fhcont)
                p = []
                v = []
                contlevels = cs.levels
                for i in range(len(cs.collections)):
                    p.append(cs.collections[i].get_paths())
                    vv = []
                    for j in range(len(p[i])):
                        vv.append(p[i][j].vertices)
                    
                    v.append(np.concatenate(vv))


                xpv = []
                ypv = []
                for i in range(len(v)):
                    xpv1 = []
                    ypv1 = []
                    for j in range(len(v[i])):
                        xpv1.append(float(v[i][j][0]))
                        ypv1.append(float(v[i][j][1]))
                    xpv.append(xpv1)
                    ypv.append(ypv1)

                lengthcontours = []
                for i in range(len(xpv)):
                    lengthcontours.append(len(xpv[i]))

                priorsample = floor((len(a.DLxx)*10)/len(xpv))
                priorsample = int(priorsample)
                print 'priorsample', priorsample

                xbath = []
                ybath = []
                depth = []
                for i in range(len(xpv)):
                    if lengthcontours[i] > priorsample:
                        xbath.append(np.random.choice(xpv[i],priorsample))
                        ybath.append(np.random.choice(ypv[i],priorsample))
                        #xbath.append(np.array(xpv[i][0::len(xpv[i])//5])) 
                        #ybath.append(np.array(ypv[i][0::len(ypv[i])//5])) 
                    else:
                        xbath.append(np.random.choice(xpv[i],lengthcontours[i]))
                        ybath.append(np.random.choice(ypv[i],lengthcontours[i]))
                        #xbath.append(np.array(xpv[i][0::len(xpv[i])//10])) 
                        #ybath.append(np.array(ypv[i][0::len(ypv[i])//10])) 
                        


                xbathh = [item for sublist in xbath for item in sublist]
                ybathh = [item for sublist in ybath for item in sublist]
                
                xbathh = np.array(xbathh)
                ybathh = np.array(ybathh)
                newrat = len(xbathh)/len(a.DLxx)
                bathsampling = floor(newrat)
                print 'bathsampling', bathsampling
                a.bathsampling = bathsampling 

                def closest(lst, K):
                    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

                xclose = []
                yclose = []
                for i in range(len(xbathh)):
                    xclose.append(closest(batt, xbathh[i]))
                    yclose.append(closest(bat, ybathh[i]))

           
                fhzip = zip(batt,bat,Dep)

                for i in range(len(fhzip)):
                    for z in range(len(xclose)):
                        if fhzip[i][0] == xclose[z]:
                            depth.append(fhzip[i][2])
                depth = np.array(depth)
                
                xclose = np.array(xclose)
                yclose = np.array(yclose)

                coordp = np.array([utm.from_latlon(i,j) for (i,j) in zip(yclose,xclose)])
                xp = np.array(map(float,coordp[:,0]))/1000
                yp = np.array(map(float,coordp[:,1]))/1000

                xx = xp 
                yy = yp  
                Depth = depth


            if ourinformation['SpillPlace'] == 'River':
                if len(bat) > (len(a.DLxx)*10):
                    Nn = len(a.DLxtv[k])*10
                    a.bathsampling = 10.0
                    nn = ceil(np.sqrt(Nn))
                    xp = bat[0::len(bat)//(int(nn)**2)-1]
                    yp = batt[0::len(batt)//(int(nn)**2)-1]
                    depth = Dep[0::len(Dep)//(int(nn)**2)-1]
                else:
                    xp = bat 
                    yp = bat 
                    depth = Dep 
                    a.bathsampling = floor(len(xp)/len(a.DLxx))
                coordp = np.array([utm.from_latlon(i,j) for (i,j) in zip(xp,yp)])
                xpp = np.array(map(float,coordp[:,0]))/1000
                ypp = np.array(map(float,coordp[:,1]))/1000

                if len(bat) > 2000:
                    xrivcont = bat[0::len(bat)//2000]
                    yrivcont = batt[0::len(batt)//2000]
                    drivcont = Dep[0::len(Dep)//2000]
                else:
                    xrivcont = bat 
                    yrivcont = batt
                    drivcont = Dep 

                xx = xpp 
                yy = ypp

            depth = np.array(depth)
            for i in range(len(depth)):
                if depth[i] > 0.0:
                    depth[i] = 0.0
            Depth = abs(depth)
            print 'Depth', Depth 
            

        if ourinformation['SpillPlace'] == 'Ocean':

            css = plt.contourf(YYa,XXa,sfield[0].reshape(len(XXa),len(YYa)))
            numcontourfs = len(css.collections)

            pf = []
            vf = []
            contflevels = css.levels
            for i in range(len(css.collections)):
                pf.append(css.collections[i].get_paths())
                vfv = []
                for j in range(len(pf[i])):
                    vfv.append(pf[i][j].vertices)
                vf.append(np.concatenate(vfv))

            contourfloc = vf[numcontourfs-1]
            xmaxcont = []
            ymaxcont = []
            for i in range(len(contourfloc)):
                xmaxcont.append(contourfloc[i][0])
                ymaxcont.append(contourfloc[i][1])
            xmaxcont = np.array(xmaxcont)
            ymaxcont = np.array(ymaxcont)

            xcc = []
            ycc = []
            for l in range(len(xmaxcont)):
                for i in range(len(v)):
                    for j in range(len(v[i])):
                        if geopy.distance.distance((ymaxcont[l],xmaxcont[l]),(ypv[i][j],xpv[i][j])).km < 2:
                            xcc.append(xpv[i])
                            ycc.append(ypv[i])

            xcc = [list(majx) for majx in set(tuple(majx) for majx in xcc)]
            ycc = [list(majx) for majx in set(tuple(majx) for majx in ycc)]

            if xcc == []:
                a.x=x
                a.y=y

                t=a.t
                
                resR = []
                for u in range(len(t)):
                    a.tt = a.newst[0][0]
                    a.ss2 = a.ss1[0][0]
                    resaR = []
                    loc = zip(xx,yy)
                    resaR=np.array(multicore21(a,loc))
                    resaR=np.transpose(resaR)
                    sum = 0
                    
                    resR.append(resaR)
                sR = np.array(resR)
                ResultP = sR[0]

                Pmaxnorm = np.array(Pmaxnorm)
                Pmaxnorm = np.ndarray.flatten(Pmaxnorm)
                Pmaxnorm = Pmaxnorm[0]

                print 'new Pmaxnorm', Pmaxnorm

                Depth = abs(Depth)
                Pbath = np.sqrt((f1*newcont)*(f2*((Depth-np.min(Depth))/(np.max(Depth)-np.min(Depth)))))
                cprior = (np.sqrt(((Depth-np.min(Depth))/(np.max(Depth)-np.min(Depth)))*(ResultP/np.max(ResultP))))*Pmaxnorm

            else:

                contconc1 = np.zeros(len(xpv))
                contconc = np.zeros(len(xpv))
                indexmax = []
                for i in range(len(xcc)):
                    indexmax.append(xpv.index(xcc[i]))
                

                ccind = []
                for j in range(len(indexmax)-1):
                    aww = indexmax[j+1]-indexmax[j]
                    ccind.append(aww)
                ccind = np.array(ccind)
                ddind = np.where(ccind>1)

                leng = max(indexmax) - min(indexmax) -2

                for i in range(len(contconc)):
                    contconc[indexmax] = 1
                    contconc1[i] = i 

                for i in range(0,min(indexmax)+1):
                    contconc1[i] = i 
                    contconc[min(indexmax)-i] = 1 - contconc1[i]/(len(xpv)-1)

                for i in range(max(indexmax)+1,len(xpv)):
                    contconc1[i] = i - max(indexmax)
                    contconc[i] = 1 - contconc1[i]/(len(xpv)-1)

                for j in range(len(ddind[0])):
                    if len(range(indexmax[ddind[0][0]+j]+1,indexmax[ddind[0][0]+j+1]))%2==0:
                        c=len(range(indexmax[ddind[0][0]+j]+1,indexmax[ddind[0][0]+j+1]))/2
                        for i in range(indexmax[ddind[0][0]+j]+1,indexmax[ddind[0][0]+j+1]):
                            if i < ((indexmax[ddind[0][0]+j]+1)+indexmax[ddind[0][0]+j+1])/2:
                                contconc1[i] = i - indexmax[ddind[0][0]+j]
                                contconc[i] = 1 - contconc1[i]/(len(xpv)-1)
                            else:
                                contconc[i] = contconc[i-c]
                    else:
                        for i in range(indexmax[ddind[0][0]+j]+1,indexmax[ddind[0][0]+j+1]):
                            if i <= ((indexmax[ddind[0][0]+j]+1)+indexmax[ddind[0][0]+j+1])/2: 
                                contconc1[i] = i - indexmax[ddind[0][0]+j]
                                contconc[i] = 1 - contconc1[i]/(len(xpv)-1)
                            else:
                                aw = int(ceil(((indexmax[ddind[0][0]+j]+1)+indexmax[ddind[0][0]+j+1])/2))
                                contconc[i] = contconc[2*aw-i]


                newcont = []
                for i in range(len(xbath)):
                    newcont.append(np.repeat(contconc[i],len(xbath[i])))
                newcont = np.array(newcont)
                newcont = np.concatenate(newcont,axis=None)

                a.x=x
                a.y=y

                t=a.t
                
                resR = []
                for u in range(len(t)):
                    a.tt = a.newst[0][0]
                    a.ss2 = a.ss1[0][0]
                    resaR = []
                    loc = zip(xx,yy)
                    resaR=np.array(multicore21(a,loc))
                    resaR=np.transpose(resaR)
                    sum = 0
                    
                    resR.append(resaR)
                sR = np.array(resR)
                ResultP = sR[0]

                Pmaxnorm = np.array(Pmaxnorm)
                Pmaxnorm = np.ndarray.flatten(Pmaxnorm)
                Pmaxnorm = Pmaxnorm[0]
                f1 = 0.7
                f2 = 0.3
                print 'new Pmaxnorm', Pmaxnorm

                Depth = abs(Depth)
                Pbath = np.sqrt((f1*newcont)*(f2*((Depth-np.min(Depth))/(np.max(Depth)-np.min(Depth)))))
                cprior = (np.sqrt((Pbath*(ResultP/np.max(ResultP)))))*Pmaxnorm
        
        if ourinformation['SpillPlace'] == 'River':
            a.x=x
            a.y=y
            t=[a.t]
            
             
            
            resR = []
            for u in range(len(t)):
                a.tt = a.newst[0][0]
                a.ss2 = a.ss1[0][0]
                resaR = []
                loc = zip(xx,yy)
                resaR=np.array(multicore21(a,loc))
                resaR=np.transpose(resaR)
                sum = 0
                resR.append(resaR)
            sR = np.array(resR)
            ResultP = sR[0]

            Pmaxnorm = np.array(Pmaxnorm)
            Pmaxnorm = np.ndarray.flatten(Pmaxnorm)
            Pmaxnorm = Pmaxnorm[0]
            print 'new Pmaxnorm', Pmaxnorm

            cprior = (np.sqrt((((Depth-np.min(Depth))/(np.max(Depth)-np.min(Depth)))*(ResultP/np.max(ResultP)))))*Pmaxnorm


        Cprior = []
        for s in cprior:
            if s == 0.0:
                conValue = (np.min(ResultP)*1e-10)
            else:
                conValue = (s)
            Cprior.append(conValue)
        Cprior = np.array(Cprior)
        
        print 'Cprior', Cprior 

        a.xx=xx
        a.yy=yy
        a.pt=pt

        a.Cprior=Cprior
        a.Cprior=a.Cprior
        a.Depth = Depth

        if ourinformation['Run'] == 'Run':
            maxlog, probNew, MaxLikelihoodFandP = LikelihoodNew(a,N)
            newLikemle = np.argmax(probNew)

            a.MaxLogLikeP = MaxLikelihoodFandP
            print a.MaxLogLikeP
            MaxLogP.append(a.MaxLogLikeP)
            print 'MaxLog', MaxLog
            print 'MaxLogP', MaxLogP
            if MaxLog[k] == MaxLogP[k]:
                a.newr = a.r 
            else:
                a.newr = parameter[newLikemle]
            
            print 'MLE parameter', a.newr
            combineparanew.append(a.newr)
        if ourinformation['Run'] == 'Recalc':
            if len(a.DLxtv) == 1:
                MaxLogP = [np.loadtxt('MaxLogLikeP.txt',delimiter=',')]
                paramsload2 = [np.loadtxt('fieldpriorparameters.txt',delimiter=',')]
                a.newr = [paramsload2[k]]
                a.combineparanew = [paramsload2[k]]
                combineparanew = [paramsload2[k]]
            else:
                MaxLogP = np.loadtxt('MaxLogLikeP.txt',delimiter=',')
                paramsload2 = np.loadtxt('fieldpriorparameters.txt',delimiter=',')
                a.newr = paramsload2[k]
                a.combineparanew = paramsload2[k]
                combineparanew = paramsload2

            MaxLogP = MaxLogP[k]


        (vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4) = combineparanew[k]

        a.xx01 = a.xx01 + vx1*a.st
        a.yy01 = a.yy01 + vy1*a.st
        a.xx02 = a.xx02 + vx2*a.st
        a.yy02 = a.yy02 + vy2*a.st
        a.xx03 = a.xx03 + vx3*a.st
        a.yy03 = a.yy03 + vy3*a.st
        a.xx04 = a.xx04 + vx4*a.st
        a.yy04 = a.yy04 + vy4*a.st   

        a.ssigmax01 = a.ssigmax01 + np.sqrt(2*Dx1*a.st)
        a.ssigmay01 = a.ssigmay01 + np.sqrt(2*Dy1*a.st)
        a.ssigmax02 = a.ssigmax02 + np.sqrt(2*Dx2*a.st)
        a.ssigmay02 = a.ssigmay02 + np.sqrt(2*Dy2*a.st)
        a.ssigmax03 = a.ssigmax03 + np.sqrt(2*Dx3*a.st)
        a.ssigmay03 = a.ssigmay03 + np.sqrt(2*Dy3*a.st)                 
        a.ssigmax04 = a.ssigmax04 + np.sqrt(2*Dx4*a.st)
        a.ssigmay04 = a.ssigmay04 + np.sqrt(2*Dy4*a.st)

    print 'last xx01', a.xx01 

    if ourinformation['Run'] == 'Run':
        np.savetxt('fieldparameters.txt',combinepara,delimiter=',')
        np.savetxt('fieldpriorparameters.txt',combineparanew,delimiter=',')
        np.savetxt('MaxLogLike.txt',MaxLog,delimiter=',')
        np.savetxt('MaxLogLikeP.txt',MaxLogP,delimiter=',')


    if ourinformation['starttime'] == ourinformation['endtime']:
        Resu = []
        pret = []
        x=a.x
        y=a.y 
        for u in range(len(t)):

            a.tt = t[u]
            a.ss2 = ss2[u]
            resaa = []
            diff = []
            diff = a.tt - a.uniST
            index = np.where(diff>=0)
            a.newr = combineparanew[np.min(index):np.max(index)+1]
            timed=diff[diff>=0]

            if diff[-1] >0:
                a.tt = timed[-1]
            if diff[-1]<=0:
                a.tt = timed[-1]
                                              
            for i in range(len(a.newr)):
                if i < len(a.newr)-1:
                    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.newr[i]                  
                if i >= len(a.newr)-1:
                    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.newr[-1]                               
                
                if i == 0: 
                    a.xx01 = a.x01 + vx1*stpredict[i]
                    a.yy01 = a.y01 + vy1*stpredict[i]
                    a.xx02 = a.x02 + vx2*stpredict[i]
                    a.yy02 = a.y02 + vy2*stpredict[i]
                    a.xx03 = a.x03 + vx3*stpredict[i]
                    a.yy03 = a.y03 + vy3*stpredict[i]
                    a.xx04 = a.x04 + vx4*stpredict[i]
                    a.yy04 = a.y04 + vy4*stpredict[i]
                    a.ssigmax01 = 0.05 + np.sqrt(2*Dx1*(stpredict[i]))
                    a.ssigmay01 = 0.05 + np.sqrt(2*Dy1*(stpredict[i]))
                    a.ssigmax02 = 0.05 + np.sqrt(2*Dx2*(stpredict[i]))
                    a.ssigmay02 = 0.05 + np.sqrt(2*Dy2*(stpredict[i]))
                    a.ssigmax03 = 0.05 + np.sqrt(2*Dx3*(stpredict[i]))
                    a.ssigmay03 = 0.05 + np.sqrt(2*Dy3*(stpredict[i]))
                    a.ssigmax04 = 0.05 + np.sqrt(2*Dx4*(stpredict[i]))
                    a.ssigmay04 = 0.05 + np.sqrt(2*Dy4*(stpredict[i]))
                else: 
                    a.xx01 = a.xx01 + vx1*stpredict[i]
                    a.yy01 = a.yy01 + vy1*stpredict[i]
                    a.xx02 = a.xx02 + vx2*stpredict[i]
                    a.yy02 = a.yy02 + vy2*stpredict[i]
                    a.xx03 = a.xx03 + vx3*stpredict[i]
                    a.yy03 = a.yy03 + vy3*stpredict[i]
                    a.xx04 = a.xx04 + vx4*stpredict[i]
                    a.yy04 = a.yy04 + vy4*stpredict[i]                    
                    a.ssigmax01 = a.ssigmax01 + np.sqrt(2*Dx1*(stpredict[i]))
                    a.ssigmay01 = a.ssigmay01 + np.sqrt(2*Dy1*(stpredict[i]))
                    a.ssigmax02 = a.ssigmax02 + np.sqrt(2*Dx2*(stpredict[i]))
                    a.ssigmay02 = a.ssigmay02 + np.sqrt(2*Dy2*(stpredict[i]))
                    a.ssigmax03 = a.ssigmax03 + np.sqrt(2*Dx3*(stpredict[i]))
                    a.ssigmay03 = a.ssigmay03 + np.sqrt(2*Dy3*(stpredict[i]))
                    a.ssigmax04 = a.ssigmax04 + np.sqrt(2*Dx4*(stpredict[i]))
                    a.ssigmay04 = a.ssigmay04 + np.sqrt(2*Dy4*(stpredict[i])) 
            

            a.tt = [a.tt]
            a.xx01 = [a.xx01]
            a.xx02 = [a.xx02]
            a.xx03 = [a.xx03]
            a.xx04 = [a.xx04]
            a.yy01 = [a.yy01]
            a.yy02 = [a.yy02]
            a.yy03 = [a.yy03]
            a.yy04 = [a.yy04]
            a.ssigmax01 = [a.ssigmax01]
            a.ssigmax02 = [a.ssigmax02]
            a.ssigmax03 = [a.ssigmax03]
            a.ssigmax04 = [a.ssigmax04]
            a.ssigmay01 = [a.ssigmay01]
            a.ssigmay02 = [a.ssigmay02]
            a.ssigmay03 = [a.ssigmay03]
            a.ssigmay04 = [a.ssigmay04]


            a.ttt = a.newr[-1] 
            print 'a.tt', a.tt
            print a.newr
            print 'a.ttt', a.ttt
            loc = zip(x,y)     
            resaa=np.array(multicore10(a,loc))
            resaa=np.transpose(resaa)
            sum = 0
            Resu.append(resaa)
        Result = np.array(Resu)
        Result = Result/np.max(Result)  

    if ourinformation['starttime'] != ourinformation['endtime']:
        Resu = []
        pret = []
        x=a.x
        y=a.y 
        a.xx01n = []
        a.yy01n = []
        a.xx02n = []
        a.yy02n = []
        a.xx03n = []
        a.yy03n = []
        a.xx04n = []
        a.yy04n = []
        a.ssigmax01n = []
        a.ssigmay01n = []
        a.ssigmax02n = []
        a.ssigmay02n = []
        a.ssigmax03n = []
        a.ssigmay03n = []
        a.ssigmax04n = []
        a.ssigmay04n = []
        a.sspre = []
        a.predt = []
        for u in range(len(t)):

            a.tt = t[u]
            a.ss2 = ss2[u]
            ss2last = a.ss2[-1]
            resaa = []
            diff = []
            diff = a.tt - a.uniST
            diffss2 = ss2last - a.uniST

            index = np.where(diff>=0)
            a.newr = combineparanew[np.min(index):np.max(index)+1]
            indexss2 = np.where(diffss2>=0)
            timed=diff[diff>=0]
            timedss2=diffss2[diffss2>=0]

            if diff[-1] >0:
                a.tt = timed[-1]
            if diff[-1]<=0:
                a.tt = timed[-1]

            newsamp = stpredict[np.min(index):np.max(index)+1]

            
            if diffss2[-1] > 0:
                a.sstt = np.linspace(0,timedss2[-1],timedss2[-1]+1)
            if diffss2[-1]<=0:
                a.sstt = np.zeros(1)


                                              
            for i in range(len(a.newr)):
                if i < len(a.newr)-1:
                    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.newr[i]                  
                if i >= len(a.newr)-1:
                    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.newr[-1]                               
                
                if i != len(a.newr)-1: 
                    a.xx01n.append(a.x01+vx1*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][0]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.xx02n.append(a.x02+vx2*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][1]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.xx03n.append(a.x03+vx3*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][2]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.xx04n.append(a.x04+vx4*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][3]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.yy01n.append(a.y01+vy1*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][4]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.yy02n.append(a.y02+vy2*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][5]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.yy03n.append(a.y03+vy3*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][6]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.yy04n.append(a.y04+vy4*(newsamp[i]-a.ss1[i][0])+a.newr[i+1][7]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                    a.ssigmax01n.append(0.05+np.sqrt(2*Dx1*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][8]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmax02n.append(0.05+np.sqrt(2*Dx2*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][9]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmax03n.append(0.05+np.sqrt(2*Dx3*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][10]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmax04n.append(0.05+np.sqrt(2*Dx4*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][11]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmay01n.append(0.05+np.sqrt(2*Dy1*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][12]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmay02n.append(0.05+np.sqrt(2*Dy2*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][13]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmay03n.append(0.05+np.sqrt(2*Dy3*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][14]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                    a.ssigmay04n.append(0.05+np.sqrt(2*Dy4*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.newr[i+1][15]*(newsamp[i+1]-a.ss1[i+1][0][0])))


                else: 
                    a.xx01n.append(a.x01+vx1*(newsamp[i]-a.ss1[i][0]))
                    a.xx02n.append(a.x01+vx2*(newsamp[i]-a.ss1[i][0]))
                    a.xx03n.append(a.x01+vx3*(newsamp[i]-a.ss1[i][0]))
                    a.xx04n.append(a.x01+vx4*(newsamp[i]-a.ss1[i][0]))
                    a.yy01n.append(a.x01+vy1*(newsamp[i]-a.ss1[i][0]))
                    a.yy02n.append(a.x01+vy2*(newsamp[i]-a.ss1[i][0]))
                    a.yy03n.append(a.x01+vy3*(newsamp[i]-a.ss1[i][0]))
                    a.yy04n.append(a.x01+vy4*(newsamp[i]-a.ss1[i][0]))
                    a.ssigmax01n.append(0.05+np.sqrt(2*Dx1*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmax02n.append(0.05+np.sqrt(2*Dx2*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmax03n.append(0.05+np.sqrt(2*Dx3*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmax04n.append(0.05+np.sqrt(2*Dx4*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmay01n.append(0.05+np.sqrt(2*Dy1*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmay02n.append(0.05+np.sqrt(2*Dy2*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmay03n.append(0.05+np.sqrt(2*Dy3*(newsamp[i]-a.ss1[i][0])))
                    a.ssigmay04n.append(0.05+np.sqrt(2*Dy4*(newsamp[i]-a.ss1[i][0])))

                    

            a.xx01n = [val for sublist in a.xx01n for val in sublist]
            a.xx02n = [val for sublist in a.xx02n for val in sublist]
            a.xx03n = [val for sublist in a.xx03n for val in sublist]
            a.xx04n = [val for sublist in a.xx04n for val in sublist]
            a.yy01n = [val for sublist in a.yy01n for val in sublist]
            a.yy02n = [val for sublist in a.yy02n for val in sublist]
            a.yy03n = [val for sublist in a.yy03n for val in sublist]
            a.yy04n = [val for sublist in a.yy04n for val in sublist]
            a.ssigmax01n = [val for sublist in a.ssigmax01n for val in sublist]
            a.ssigmax02n = [val for sublist in a.ssigmax02n for val in sublist]
            a.ssigmax03n = [val for sublist in a.ssigmax03n for val in sublist]
            a.ssigmax04n = [val for sublist in a.ssigmax04n for val in sublist]
            a.ssigmay01n = [val for sublist in a.ssigmay01n for val in sublist]
            a.ssigmay02n = [val for sublist in a.ssigmay02n for val in sublist]
            a.ssigmay03n = [val for sublist in a.ssigmay03n for val in sublist]
            a.ssigmay04n = [val for sublist in a.ssigmay04n for val in sublist]

            a.xx01n = np.array(a.xx01n)
            a.xx02n = np.array(a.xx02n)
            a.xx03n = np.array(a.xx03n)
            a.xx04n = np.array(a.xx04n)
            a.yy01n = np.array(a.yy01n)
            a.yy02n = np.array(a.yy02n)
            a.yy03n = np.array(a.yy03n)
            a.yy04n = np.array(a.yy04n)
            a.ssigmax01n = np.array(a.ssigmax01n)
            a.ssigmax02n = np.array(a.ssigmax02n)
            a.ssigmax03n = np.array(a.ssigmax03n)
            a.ssigmax04n = np.array(a.ssigmax04n)
            a.ssigmay01n = np.array(a.ssigmay01n)
            a.ssigmay02n = np.array(a.ssigmay02n)
            a.ssigmay03n = np.array(a.ssigmay03n)
            a.ssigmay04n = np.array(a.ssigmay04n)
            
            a.ss2_new = np.zeros(len(a.xx01n)-1)
            a.ss2_new = np.append(a.ss2_new,a.tt)
            a.ss2 = np.append(a.ss2_new,a.sstt)

            x01new = a.x01*np.ones(len(a.sstt))
            x02new = a.x02*np.ones(len(a.sstt))
            x03new = a.x03*np.ones(len(a.sstt))
            x04new = a.x04*np.ones(len(a.sstt))
            y01new = a.y01*np.ones(len(a.sstt))
            y02new = a.y02*np.ones(len(a.sstt))
            y03new = a.y03*np.ones(len(a.sstt))
            y04new = a.y04*np.ones(len(a.sstt))
            ssx01new = ssx02new = ssx03new = ssx04new = ssy01new = ssy02new = ssy03new = ssy04new = 0.05*np.ones(len(a.sstt))

            a.xx01 = np.append(a.xx01n,x01new)
            a.xx02 = np.append(a.xx02n,x02new)
            a.xx03 = np.append(a.xx03n,x03new)
            a.xx04 = np.append(a.xx04n,x04new)
            a.yy01 = np.append(a.yy01n,y01new)
            a.yy02 = np.append(a.yy02n,y02new)
            a.yy03 = np.append(a.yy03n,y03new)
            a.yy04 = np.append(a.yy04n,y04new)
            a.ssigmax01 = np.append(a.ssigmax01n,ssx01new)
            a.ssigmax02 = np.append(a.ssigmax02n,ssx02new)
            a.ssigmax03 = np.append(a.ssigmax03n,ssx03new)
            a.ssigmax04 = np.append(a.ssigmax04n,ssx04new)
            a.ssigmay01 = np.append(a.ssigmay01n,ssy01new)
            a.ssigmay02 = np.append(a.ssigmay02n,ssy02new)
            a.ssigmay03 = np.append(a.ssigmay03n,ssy03new)
            a.ssigmay04 = np.append(a.ssigmay04n,ssy04new)

            a.tt = a.tt*np.ones(len(a.xx01))


            a.ttt = a.newr[-1] 
            print 'a.tt', a.tt
            print a.newr
            print 'a.ttt', a.ttt
            loc = zip(x,y)     
            resaa=np.array(multicore10(a,loc))
            resaa=np.transpose(resaa)
            sum = 0
            Resu.append(resaa)
        Result = np.array(Resu)
        Result = Result/np.max(Result) 
                    

            

   
    progressBar.setValue(50)

    ###################################################     CONFIDENCE BOUND CALCULATION        ########################################################
    #___________________________________________________________________________________________________________________________________________________

    if ourinformation['Method'] == 'Minimum':  
        bounds1=[(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax']))] 
        bounds2=[(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax']))]
        bounds3=[(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax']))]
        bounds4=[(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax']))]
        bounds5=[(-0.999,0.999),(-0.999,0.999),(-0.999,0.999),(-0.999,0.999)]
        a.par = []
        rescf = []

        
        for k in range(len(a.DLxtv)):
            a.DLx_t = a.DLxtv[k]
            a.DLy_t = a.DLytv[k]
            a.DLcon_t = a.DLcontv[k]
            a.ss1_tcf = a.ss1[k]
            a.DLxx = a.DLx_t
            a.DLyy = a.DLy_t
            a.DLconny = a.DLcon_t
            if k == 0: 
                a.st = a.newst[k][0]
            else: 
                a.st = a.newst[k][0]-a.newst[k-1][0]
            tcf = a.st
            a.rr = combineparanew[k]
            a.MaxLogLike = MaxLog[k]
            a.MaxLogLikeP = MaxLogP[k]
            if k == 0:
                a.xx01 = a.x01
                a.yy01 = a.y01
                a.xx02 = a.x02
                a.yy02 = a.y02
                a.xx03 = a.x03
                a.yy03 = a.y03
                a.xx04 = a.x04
                a.yy04 = a.y04     
                a.ssigmax01 = 0.05 
                a.ssigmay01 = 0.05                   
                a.ssigmax02 = 0.05 
                a.ssigmay02 = 0.05 
                a.ssigmax03 = 0.05 
                a.ssigmay03 = 0.05               
                a.ssigmax04 = 0.05 
                a.ssigmay04 = 0.05  
                result = differential_evolution(partial(IniLikelihood1,a),bounds1,seed=0,maxiter=10000)
                print result

                fitted_params = result.x
                print(fitted_params)
                
                vx1 = fitted_params[0]
                vx2 = fitted_params[1]
                vx3 = fitted_params[2]
                vx4 = fitted_params[3]

                a.fitted_params = vx1,vx2,vx3,vx4 
                print a.fitted_params


                    
                result2 = differential_evolution(partial(IniLikelihood2,a),bounds2,seed=0,maxiter=10000)
                print result2
                fitted_params2 = result2.x
                print(fitted_params2)
                vy1 = fitted_params2[0]
                vy2 = fitted_params2[1]
                vy3 = fitted_params2[2]
                vy4 = fitted_params2[3]


                a.fitted_params2 = vy1,vy2,vy3,vy4 
                print a.fitted_params2
   
                result3 = differential_evolution(partial(IniLikelihood3,a),bounds3,seed=0,maxiter=10000)
                print result3
                
                fitted_params3 = result3.x
                print(fitted_params3)


                Dx1 = fitted_params3[0]
                Dx2 = fitted_params3[1]
                Dx3 = fitted_params3[2]
                Dx4 = fitted_params3[3]
                a.fitted_params3 = Dx1,Dx2,Dx3,Dx4
                print a.fitted_params3

                
                result4 = differential_evolution(partial(IniLikelihood4,a),bounds4,seed=0,maxiter=10000)
                print result4
        
                fitted_params4 = result4.x
                print(fitted_params4)


                Dy1 = fitted_params4[0]
                Dy2 = fitted_params4[1]
                Dy3 = fitted_params4[2]
                Dy4 = fitted_params4[3]
                a.fitted_params4 = Dy1,Dy2,Dy3,Dy4

                print a.fitted_params4
    
                result5 = differential_evolution(partial(IniLikelihood5,a),bounds5,seed=0,maxiter=10000)
                print result5  

                fitted_params5 = result5.x
                print(fitted_params5)
 

                ro1 = fitted_params5[0]
                ro2 = fitted_params5[1]
                ro3 = fitted_params5[2]
                ro4 = fitted_params5[3]
                a.fitted_params5 = ro1,ro2,ro3,ro4
                print a.fitted_params5


                gamma1 = a.rr[20]
                gamma2 = a.rr[21]
                gamma3 = a.rr[22]
                gamma4 = a.rr[23]
                par = vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4
                a.par.append(par)

            else:

                a.xx01 = a.xx01 + vx1*tcf
                a.yy01 = a.yy01 + vy1*tcf
                a.xx02 = a.xx02 + vx2*tcf
                a.yy02 = a.yy02 + vy2*tcf
                a.xx03 = a.xx03 + vx3*tcf
                a.yy03 = a.yy03 + vy3*tcf
                a.xx04 = a.xx04 + vx4*tcf
                a.yy04 = a.yy04 + vy4*tcf
                a.ssigmax01 = a.ssigmax01 + np.sqrt(2*Dx1*tcf)
                a.ssigmay01 = a.ssigmay01 + np.sqrt(2*Dy1*tcf)                  
                a.ssigmax02 = a.ssigmax02 + np.sqrt(2*Dx2*tcf)
                a.ssigmay02 = a.ssigmay02 + np.sqrt(2*Dy2*tcf) 
                a.ssigmax03 = a.ssigmax03 + np.sqrt(2*Dx3*tcf)
                a.ssigmay03 = a.ssigmay03 + np.sqrt(2*Dy3*tcf)                  
                a.ssigmax04 = a.ssigmax04 + np.sqrt(2*Dx4*tcf)
                a.ssigmay04 = a.ssigmay04 + np.sqrt(2*Dy4*tcf)

                result = differential_evolution(partial(IniLikelihood1,a),bounds1,seed=0,maxiter=10000)
                print result

        
                fitted_params = result.x
                print(fitted_params)
                
                
                vx1 = fitted_params[0]
                vx2 = fitted_params[1]
                vx3 = fitted_params[2]
                vx4 = fitted_params[3]
                

                a.fitted_params = vx1,vx2,vx3,vx4 
                print a.fitted_params


                    
                result2 = differential_evolution(partial(IniLikelihood2,a),bounds2,seed=0,maxiter=10000)
                print result2
                
                fitted_params2 = result2.x
                print(fitted_params2)

                vy1 = fitted_params2[0]
                vy2 = fitted_params2[1]
                vy3 = fitted_params2[2]
                vy4 = fitted_params2[3]


                a.fitted_params2 = vy1,vy2,vy3,vy4 
                print a.fitted_params2
                  
                result3 = differential_evolution(partial(IniLikelihood3,a),bounds3,seed=0,maxiter=10000)    
                print result3
                
                fitted_params3 = result3.x
                print(fitted_params3)

                Dx1 = fitted_params3[0]
                Dx2 = fitted_params3[1]
                Dx3 = fitted_params3[2]
                Dx4 = fitted_params3[3]
                a.fitted_params3 = Dx1,Dx2,Dx3,Dx4
                print a.fitted_params3

                result4 = differential_evolution(partial(IniLikelihood4,a),bounds4,seed=0,maxiter=10000)
                print result4

                fitted_params4 = result4.x
                print(fitted_params4)
                

                Dy1 = fitted_params4[0]
                Dy2 = fitted_params4[1]
                Dy3 = fitted_params4[2]
                Dy4 = fitted_params4[3]
                a.fitted_params4 = Dy1,Dy2,Dy3,Dy4

                print a.fitted_params4

               
                result5 = differential_evolution(partial(IniLikelihood5,a),bounds5,seed=0,maxiter=10000)
                print result5  
                
                fitted_params5 = result5.x
                print(fitted_params5)
                

                ro1 = fitted_params5[0]
                ro2 = fitted_params5[1]
                ro3 = fitted_params5[2]
                ro4 = fitted_params5[3]
                a.fitted_params5 = ro1,ro2,ro3,ro4
                print a.fitted_params5


                gamma1 = a.rr[20]
                gamma2 = a.rr[21]
                gamma3 = a.rr[22]
                gamma4 = a.rr[23]
                par = vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4
                a.par.append(par)


        if ourinformation['starttime'] == ourinformation['endtime']:
            for u in range(len(a.t)):


                a.tt = a.t[u]
                a.ss2 = ss2[u]
                resaa = []
                diff = []
                diff = a.tt - a.uniST
                index = np.where(diff>=0)
                a.newr = combineparanew[np.min(index):np.max(index)+1]
                timed=diff[diff>=0]

                if diff[-1] >0:
                    a.tt = timed[-1]
                if diff[-1]<=0:
                    a.tt = timed[-1]       

                for i in range(len(a.par)):
                    if i < len(a.par)-1:
                        vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[i]                  
                    if i >= len(a.par)-1:
                        vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[-1]                               

                    if i == 0:
                        a.xx01 = a.x01 + vx1*stpredict[i]
                        a.yy01 = a.y01 + vy1*stpredict[i]
                        a.xx02 = a.x02 + vx2*stpredict[i]
                        a.yy02 = a.y02 + vy2*stpredict[i]
                        a.xx03 = a.x03 + vx3*stpredict[i]
                        a.yy03 = a.y03 + vy3*stpredict[i]
                        a.xx04 = a.x04 + vx4*stpredict[i]
                        a.yy04 = a.y04 + vy4*stpredict[i]
                        a.ssigmax01 = 0.05 + np.sqrt(2*Dx1*(stpredict[i]))
                        a.ssigmay01 = 0.05 + np.sqrt(2*Dy1*(stpredict[i]))
                        a.ssigmax02 = 0.05 + np.sqrt(2*Dx2*(stpredict[i]))
                        a.ssigmay02 = 0.05 + np.sqrt(2*Dy2*(stpredict[i]))
                        a.ssigmax03 = 0.05 + np.sqrt(2*Dx3*(stpredict[i]))
                        a.ssigmay03 = 0.05 + np.sqrt(2*Dy3*(stpredict[i]))
                        a.ssigmax04 = 0.05 + np.sqrt(2*Dx4*(stpredict[i]))
                        a.ssigmay04 = 0.05 + np.sqrt(2*Dy4*(stpredict[i]))
                    else: 
                        a.xx01 = a.xx01 + vx1*stpredict[i]
                        a.yy01 = a.yy01 + vy1*stpredict[i]
                        a.xx02 = a.xx02 + vx2*stpredict[i]
                        a.yy02 = a.yy02 + vy2*stpredict[i]
                        a.xx03 = a.xx03 + vx3*stpredict[i]
                        a.yy03 = a.yy03 + vy3*stpredict[i]
                        a.xx04 = a.xx04 + vx4*stpredict[i]
                        a.yy04 = a.yy04 + vy4*stpredict[i]                    
                        a.ssigmax01 = a.ssigmax01 + np.sqrt(2*Dx1*(stpredict[i]))
                        a.ssigmay01 = a.ssigmay01 + np.sqrt(2*Dy1*(stpredict[i]))
                        a.ssigmax02 = a.ssigmax02 + np.sqrt(2*Dx2*(stpredict[i]))
                        a.ssigmay02 = a.ssigmay02 + np.sqrt(2*Dy2*(stpredict[i]))
                        a.ssigmax03 = a.ssigmax03 + np.sqrt(2*Dx3*(stpredict[i]))
                        a.ssigmay03 = a.ssigmay03 + np.sqrt(2*Dy3*(stpredict[i]))
                        a.ssigmax04 = a.ssigmax04 + np.sqrt(2*Dx4*(stpredict[i]))
                        a.ssigmay04 = a.ssigmay04 + np.sqrt(2*Dy4*(stpredict[i]))
                

                a.tt = [a.tt]
                a.xx01 = [a.xx01]
                a.xx02 = [a.xx02]
                a.xx03 = [a.xx03]
                a.xx04 = [a.xx04]
                a.yy01 = [a.yy01]
                a.yy02 = [a.yy02]
                a.yy03 = [a.yy03]
                a.yy04 = [a.yy04]
                a.ssigmax01 = [a.ssigmax01]
                a.ssigmax02 = [a.ssigmax02]
                a.ssigmax03 = [a.ssigmax03]
                a.ssigmax04 = [a.ssigmax04]
                a.ssigmay01 = [a.ssigmay01]
                a.ssigmay02 = [a.ssigmay02]
                a.ssigmay03 = [a.ssigmay03]
                a.ssigmay04 = [a.ssigmay04]
                a.ppar = a.par[-1]
                loc = zip(x,y)     
                resacf=np.array(multicore5(a,loc))
                resacf=np.transpose(resacf)
                sum = 0
                rescf.append(resacf)
            scf = np.array(rescf)
            scf = scf/np.max(scf)

        if ourinformation['starttime'] != ourinformation['endtime']:
            Resu = []
            pret = []
            x=a.x
            y=a.y 
            a.xx01n = []
            a.yy01n = []
            a.xx02n = []
            a.yy02n = []
            a.xx03n = []
            a.yy03n = []
            a.xx04n = []
            a.yy04n = []
            a.ssigmax01n = []
            a.ssigmay01n = []
            a.ssigmax02n = []
            a.ssigmay02n = []
            a.ssigmax03n = []
            a.ssigmay03n = []
            a.ssigmax04n = []
            a.ssigmay04n = []
            a.sspre = []
            a.predt = []
            for u in range(len(t)):

                a.tt = t[u]
                a.ss2 = ss2[u]
                ss2last = a.ss2[-1]
                resaa = []
                diff = []
                diff = a.tt - a.uniST
                diffss2 = ss2last - a.uniST

                index = np.where(diff>=0)
                a.newr = combineparanew[np.min(index):np.max(index)+1]
                indexss2 = np.where(diffss2>=0)
                timed=diff[diff>=0]
                timedss2=diffss2[diffss2>=0]

                if diff[-1] >0:
                    a.tt = timed[-1]
                if diff[-1]<=0:
                    a.tt = timed[-1]

                newsamp = stpredict[np.min(index):np.max(index)+1]

                
                if diffss2[-1] > 0:
                    a.sstt = np.linspace(0,timedss2[-1],timedss2[-1]+1)
                if diffss2[-1]<=0:
                    a.sstt = np.zeros(1)

                                                  
                for i in range(len(a.par)):
                    if i < len(a.par)-1:
                        vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[i]                  
                    if i >= len(a.par)-1:
                        vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[-1]                               
                    
                    if i != len(a.newr)-1: 
                        a.xx01n.append(a.x01+vx1*(newsamp[i]-a.ss1[i][0])+a.par[i+1][0]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.xx02n.append(a.x02+vx2*(newsamp[i]-a.ss1[i][0])+a.par[i+1][1]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.xx03n.append(a.x03+vx3*(newsamp[i]-a.ss1[i][0])+a.par[i+1][2]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.xx04n.append(a.x04+vx4*(newsamp[i]-a.ss1[i][0])+a.par[i+1][3]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.yy01n.append(a.y01+vy1*(newsamp[i]-a.ss1[i][0])+a.par[i+1][4]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.yy02n.append(a.y02+vy2*(newsamp[i]-a.ss1[i][0])+a.par[i+1][5]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.yy03n.append(a.y03+vy3*(newsamp[i]-a.ss1[i][0])+a.par[i+1][6]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.yy04n.append(a.y04+vy4*(newsamp[i]-a.ss1[i][0])+a.par[i+1][7]*(newsamp[i+1]-a.ss1[i+1][0][0]))
                        a.ssigmax01n.append(0.05+np.sqrt(2*Dx1*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][8]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmax02n.append(0.05+np.sqrt(2*Dx2*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][9]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmax03n.append(0.05+np.sqrt(2*Dx3*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][10]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmax04n.append(0.05+np.sqrt(2*Dx4*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][11]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmay01n.append(0.05+np.sqrt(2*Dy1*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][12]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmay02n.append(0.05+np.sqrt(2*Dy2*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][13]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmay03n.append(0.05+np.sqrt(2*Dy3*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][14]*(newsamp[i+1]-a.ss1[i+1][0][0])))
                        a.ssigmay04n.append(0.05+np.sqrt(2*Dy4*(newsamp[i]-a.ss1[i][0]))+np.sqrt(2*a.par[i+1][15]*(newsamp[i+1]-a.ss1[i+1][0][0])))


                    else: 
                        a.xx01n.append(a.x01+vx1*(newsamp[i]-a.ss1[i][0]))
                        a.xx02n.append(a.x01+vx2*(newsamp[i]-a.ss1[i][0]))
                        a.xx03n.append(a.x01+vx3*(newsamp[i]-a.ss1[i][0]))
                        a.xx04n.append(a.x01+vx4*(newsamp[i]-a.ss1[i][0]))
                        a.yy01n.append(a.x01+vy1*(newsamp[i]-a.ss1[i][0]))
                        a.yy02n.append(a.x01+vy2*(newsamp[i]-a.ss1[i][0]))
                        a.yy03n.append(a.x01+vy3*(newsamp[i]-a.ss1[i][0]))
                        a.yy04n.append(a.x01+vy4*(newsamp[i]-a.ss1[i][0]))
                        a.ssigmax01n.append(0.05+np.sqrt(2*Dx1*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmax02n.append(0.05+np.sqrt(2*Dx2*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmax03n.append(0.05+np.sqrt(2*Dx3*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmax04n.append(0.05+np.sqrt(2*Dx4*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmay01n.append(0.05+np.sqrt(2*Dy1*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmay02n.append(0.05+np.sqrt(2*Dy2*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmay03n.append(0.05+np.sqrt(2*Dy3*(newsamp[i]-a.ss1[i][0])))
                        a.ssigmay04n.append(0.05+np.sqrt(2*Dy4*(newsamp[i]-a.ss1[i][0])))

                        

                a.xx01n = [val for sublist in a.xx01n for val in sublist]
                a.xx02n = [val for sublist in a.xx02n for val in sublist]
                a.xx03n = [val for sublist in a.xx03n for val in sublist]
                a.xx04n = [val for sublist in a.xx04n for val in sublist]
                a.yy01n = [val for sublist in a.yy01n for val in sublist]
                a.yy02n = [val for sublist in a.yy02n for val in sublist]
                a.yy03n = [val for sublist in a.yy03n for val in sublist]
                a.yy04n = [val for sublist in a.yy04n for val in sublist]
                a.ssigmax01n = [val for sublist in a.ssigmax01n for val in sublist]
                a.ssigmax02n = [val for sublist in a.ssigmax02n for val in sublist]
                a.ssigmax03n = [val for sublist in a.ssigmax03n for val in sublist]
                a.ssigmax04n = [val for sublist in a.ssigmax04n for val in sublist]
                a.ssigmay01n = [val for sublist in a.ssigmay01n for val in sublist]
                a.ssigmay02n = [val for sublist in a.ssigmay02n for val in sublist]
                a.ssigmay03n = [val for sublist in a.ssigmay03n for val in sublist]
                a.ssigmay04n = [val for sublist in a.ssigmay04n for val in sublist]

                a.xx01n = np.array(a.xx01n)
                a.xx02n = np.array(a.xx02n)
                a.xx03n = np.array(a.xx03n)
                a.xx04n = np.array(a.xx04n)
                a.yy01n = np.array(a.yy01n)
                a.yy02n = np.array(a.yy02n)
                a.yy03n = np.array(a.yy03n)
                a.yy04n = np.array(a.yy04n)
                a.ssigmax01n = np.array(a.ssigmax01n)
                a.ssigmax02n = np.array(a.ssigmax02n)
                a.ssigmax03n = np.array(a.ssigmax03n)
                a.ssigmax04n = np.array(a.ssigmax04n)
                a.ssigmay01n = np.array(a.ssigmay01n)
                a.ssigmay02n = np.array(a.ssigmay02n)
                a.ssigmay03n = np.array(a.ssigmay03n)
                a.ssigmay04n = np.array(a.ssigmay04n)
                
                a.ss2_new = np.zeros(len(a.xx01n)-1)
                a.ss2_new = np.append(a.ss2_new,a.tt)
                a.ss2 = np.append(a.ss2_new,a.sstt)

                x01new = a.x01*np.ones(len(a.sstt))
                x02new = a.x02*np.ones(len(a.sstt))
                x03new = a.x03*np.ones(len(a.sstt))
                x04new = a.x04*np.ones(len(a.sstt))
                y01new = a.y01*np.ones(len(a.sstt))
                y02new = a.y02*np.ones(len(a.sstt))
                y03new = a.y03*np.ones(len(a.sstt))
                y04new = a.y04*np.ones(len(a.sstt))
                ssx01new = ssx02new = ssx03new = ssx04new = ssy01new = ssy02new = ssy03new = ssy04new = 0.05*np.ones(len(a.sstt))

                a.xx01 = np.append(a.xx01n,x01new)
                a.xx02 = np.append(a.xx02n,x02new)
                a.xx03 = np.append(a.xx03n,x03new)
                a.xx04 = np.append(a.xx04n,x04new)
                a.yy01 = np.append(a.yy01n,y01new)
                a.yy02 = np.append(a.yy02n,y02new)
                a.yy03 = np.append(a.yy03n,y03new)
                a.yy04 = np.append(a.yy04n,y04new)
                a.ssigmax01 = np.append(a.ssigmax01n,ssx01new)
                a.ssigmax02 = np.append(a.ssigmax02n,ssx02new)
                a.ssigmax03 = np.append(a.ssigmax03n,ssx03new)
                a.ssigmax04 = np.append(a.ssigmax04n,ssx04new)
                a.ssigmay01 = np.append(a.ssigmay01n,ssy01new)
                a.ssigmay02 = np.append(a.ssigmay02n,ssy02new)
                a.ssigmay03 = np.append(a.ssigmay03n,ssy03new)
                a.ssigmay04 = np.append(a.ssigmay04n,ssy04new)

                a.tt = a.tt*np.ones(len(a.xx01))
                a.ppar = a.par[-1]
                loc = zip(x,y)     
                resacf=np.array(multicore5(a,loc))
                resacf=np.transpose(resacf)
                sum = 0
                rescf.append(resacf)
            scf = np.array(rescf)
            scf = scf/np.max(scf)


        

    elif ourinformation['Method'] == 'Best':
        pass 


    progressBar.setValue(75)
    
    coni = [n*10 for n in a.DLcon]

    ################################################ CONVERTING OUTPUT TO PLOT IN KM (DISTANCE) ########################################################
    #___________________________________________________________________________________________________________________________________________________

    if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
        if ourinformation['SpillPlace'] == 'Ocean':
            cccord = zip(bat,batt)
            ccoordcon = zip(cccord,fhcont)
            datab = sorted(ccoordcon,key=extract_key)
            resultb = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datab, extract_key)]

            fb = []
            fh = []
            for i in range(len(resultb)):
                conresb = [list(p) for m, p in itertools.groupby(resultb[i][1],lambda x:x[0])]
                for j in conresb:
                    fh.append(j[0][0])
                    h = resultb[i][0]
                    fb.append(h)

            latcont = []
            longcont = []
            for i in range(len(fb)):
                l=fb[i][0]
                t=fb[i][1]
                latcont.append(l)
                longcont.append(t)
            xkmbat = latcont 
            ykmbat = longcont

            xaxisbat = []
            yaxisbat = []
            newxxbat = np.ones([len(xkmbat)])
            for i in range(len(xkmbat)):
                newxxbat[i] = geopy.distance.distance((Xa[0],Ya[0]),(xkmbat[i],Ya[0])).km
                xaxisbat.append(newxxbat[i])
            newyybat = np.ones([len(xkmbat)])
            for i in range(len(ykmbat)):
                newyybat[i] = geopy.distance.distance((Xa[0],Ya[0]),(Xa[0],ykmbat[i])).km
                yaxisbat.append(newyybat[i])

        if ourinformation['SpillPlace'] == 'River':
            cccord = zip(xrivcont,yrivcont)
            ccoordcon = zip(cccord,drivcont)
            datab = sorted(ccoordcon,key=extract_key)
            resultb = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datab, extract_key)]

            fb = []
            fh = []
            for i in range(len(resultb)):
                conresb = [list(p) for m, p in itertools.groupby(resultb[i][1],lambda x:x[0])]
                for j in conresb:
                    fh.append(j[0][0])
                    h = resultb[i][0]
                    fb.append(h)

            latcont = []
            longcont = []
            for i in range(len(fb)):
                l=fb[i][0]
                t=fb[i][1]
                latcont.append(l)
                longcont.append(t)
            xkmbat = latcont 
            ykmbat = longcont

            xaxisbat = []
            yaxisbat = []
            newxxbat = np.ones([len(xkmbat)])
            for i in range(len(xkmbat)):
                newxxbat[i] = geopy.distance.distance((Xa[0],Ya[0]),(xkmbat[i],Ya[0])).km
                xaxisbat.append(newxxbat[i])
            newyybat = np.ones([len(xkmbat)])
            for i in range(len(ykmbat)):
                newyybat[i] = geopy.distance.distance((Xa[0],Ya[0]),(Xa[0],ykmbat[i])).km
                yaxisbat.append(newyybat[i])

    xkm = Xa 
    ykm = Ya
    
    xaxis = []
    
    newxx = np.ones([len(xkm)])
    for i in range(len(xkm)):
        newxx[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[i],ykm[0])).km
        xaxis.append(newxx[i])

    yaxis = []
    
    newyy = np.ones([len(xkm)])
    for i in range(len(ykm)):
        newyy[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],ykm[i])).km
        yaxis.append(newyy[i])


    newxkm = newx 
    newykm = newy
    newxaxis = []
    nnewx = np.ones([len(newxkm)])
    for i in range(len(newxkm)):
        nnewx[i] = geopy.distance.distance((newxkm[0],newykm[0]),(newxkm[i],newykm[0])).km
        newxaxis.append(nnewx[i])
    
    newyaxis = []
    nnewy = np.ones([len(newykm)])
    for i in range(len(newykm)):
        nnewy[i] = geopy.distance.distance((newxkm[0],newykm[0]),(newxkm[0],newykm[i])).km
        newyaxis.append(nnewy[i])

    SXkm = []
    newxfield = np.ones([len(a.lati)])
    for i in range(len(a.longi)):
        newxfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],a.longi[i])).km 
        SXkm.append(newxfield[i])
    
    SYkm = []
    newyfield = np.ones([len(a.lati)])
    for i in range(len(a.longi)):
        newyfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(a.lati[i],ykm[0])).km 
        SYkm.append(newyfield[i])

    lon0km = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],lon0)).km 
    lat0km = geopy.distance.distance((xkm[0],ykm[0]),(lat0,ykm[0])).km 

    xcont = Xa
    ycont = Ya
    print ourinformation['SpillPlace']

    db = oceansdb.ETOPO()
    dcont = db['topography'].extract(lat=xcont, lon=ycont)
    decont = dcont['height']

    [ycdf,xcdf] = np.meshgrid(ycont,xcont)
    loccdf = []
    for i in range(len(decont)):
        loccdf.append(zip(xcdf[i],ycdf[i],decont[i]))
    newxcdf = []
    newycdf = []
    newdcdf = []
    for i in range(len(loccdf)):
        for j in range(len(loccdf[i])):
            newxcdf.append(loccdf[i][j][0])
            newycdf.append(loccdf[i][j][1])
            newdcdf.append(loccdf[i][j][2])
    newxcdf = np.array(newxcdf)
    newycdf = np.array(newycdf)
    newdcdf = np.array(newdcdf)
    
    labels = []
    t=[a.t]

    [xcdf,ycdf] = np.meshgrid(Xa,Ya)
    xcdf = np.concatenate(xcdf)
    ycdf = np.concatenate(ycdf)



    ################################################ HOW TO CREATE A NETCDF FILE FOR THE OUTPUT ########################################################
    #___________________________________________________________________________________________________________________________________________________

    # f = nc4.Dataset('output-DBL152.nc','w', format='NETCDF4')



    # f.createDimension('lon', len(ycdf))
    # f.createDimension('lat', len(xcdf))
    # f.createDimension('Rc', len(Result[0]))
    # f.createDimension('fdlon', len(a.longi))
    # f.createDimension('fdlat', len(a.lati))
    # f.createDimension('fdc', len(a.DLcon))
    # f.createDimension('cllon', len(newycdf))
    # f.createDimension('cllat', len(newxcdf))
    # f.createDimension('cldepth', len(newdcdf))
    # f.createDimension('cblon', len(y))
    # f.createDimension('cblat', len(x))
    # f.createDimension('cbcon', len(scf[0]))

    # longitude = f.createVariable('Longitude', 'f4', ('lon',))
    # latitude = f.createVariable('Latitude', 'f4', ('lat',))
    # Relative_conc = f.createVariable('Relative_concentration', 'f4', ('Rc',))
    # Field_lon = f.createVariable('Field_Data_Longitude', 'f4', ('fdlon',))
    # Field_lat = f.createVariable('Field_Data_Latitude', 'f4', ('fdlat',))
    # Field_conc = f.createVariable('Field_Data_concentration', 'f4', ('fdc',))
    # Contour_lon = f.createVariable('Contour_Line_Longitude', 'f4', ('cllon',))
    # Contour_lat = f.createVariable('Contour_Line_Latitude', 'f4', ('cllat',))
    # Contour_depth = f.createVariable('Contour_Line_depth', 'f4', ('cldepth',))
    # Conf_lon = f.createVariable('Confidence_Bound_Longitude', 'f4', ('cblon',))
    # Conf_lat = f.createVariable('Confidence_Bound_Latitude', 'f4', ('cblat',))
    # Conf_con = f.createVariable('Confidence_Bound_Rel_Concentration', 'f4', ('cbcon',))

    # longitude[:] = ycdf 
    # latitude[:] = xcdf 
    # Relative_conc[:] = Result[0] 
    # Field_lon[:] = a.longi 
    # Field_lat[:] = a.lati 
    # Field_conc[:] = a.DLcon 
    # Contour_lon[:] = newycdf
    # Contour_lat[:] = newxcdf 
    # Contour_depth[:] = newdcdf
    # Conf_lon[:] = ycdf
    # Conf_lat[:] = xcdf
    # Conf_con[:] = scf[0] 

    # f.description = "NetCDF file for the output figure for the DBL-152 oil spill. Prediction time = Sampling time"
    # today = datetime.datetime.today()
    # f.history = "Created " + today.strftime("%d/%m/%y")

    # longitude.units = 'degrees west'
    # latitude.units = 'degrees north'
    # Relative_conc.units = 'Relative Concentration'
    # Field_lon.units = 'degrees west'
    # Field_lat.units = 'degrees north'
    # Field_conc.units = 'Concentration'
    # Contour_lon.units = 'degrees west'
    # Contour_lat.units = 'degrees north'
    # Contour_depth.units = 'meters below surface'
    # Conf_lon.units = 'degrees west'
    # Conf_lat.units = 'degrees north'
    # Conf_con.units = 'Relative Concentration'

    # f.close()

############################################################## PLOTTING THE RESULT ######################################################################
    #___________________________________________________________________________________________________________________________________________________


    #print Result
    for i in range(len(t)):
        # plt.clf()
        plt.figure()
        plt.rcParams['font.size'] = 10   # change the font size of colorbar
        #plt.rcParams['font.weight'] = 'bold' # make the test bolder        
        # print "xa",Xa
        # print "Ya",Ya
        if ourinformation['Method'] == 'Best':
            if ourinformation['Map'] == 'Coordinate':
                if ourinformation['SpillPlace'] == 'Ocean':
                    plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newy,newx,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                    ax = plt.gca()
                    ax.set_facecolor('xkcd:royal')
                #plt.xticks(rotation=90)
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                plt.colorbar()
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'  
                cs3=plt.plot(a.lon0,a.lat0,ms=10,c='r',marker='+',label='Spill_Location')
                plt.legend([cs3],['Spill_Location'], loc='lower left',ncol=4, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))               
                if ourinformation['Plot'] =="nofield":
                    pass
                elif ourinformation['Plot']=="field":
                    plt.scatter(a.longi,a.lati,s=coni,label="Field_Data",color='blue')
                    plt.legend(scatterpoints=1, frameon=False, labelspacing=1,ncol=4,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                                     
                if ourinformation['contour'] =="nocontour":
                    pass                
                elif ourinformation['contour'] == 'contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                            #plt.clabel(cs, inline=0.5,fontsize=8)
                            plt.clabel(cs2, inline=0.5,fontsize=8)
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(batt,bat,Dep,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.tricontour(batt,bat,fhcont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                            #plt.clabel(cs, inline=0.5,fontsize=8)
                            plt.clabel(cs2, inline=0.5,fontsize=8)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newy,newx,pdepth,cmap=plt.get_cmap('hot'), linewidths=1)
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yrivcont,xrivcont,drivcont,cmap=plt.get_cmap('hot'), linewidths=1)
                        #cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                    level=cs2.levels
                    for i in range(len(level)):
                        lab = str(level[i]) + ' m'
                        labels.append(lab)
                    for i in range(len(labels)):
                        cs2.collections[i].set_label(labels[i])
                    plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='small',ncol=4, bbox_to_anchor=(0,1.02,1,0.2))                         

            elif ourinformation['Map'] == 'km':
                if ourinformation['SpillPlace'] == 'Ocean': 
                    plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newxaxis,newyaxis,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                plt.colorbar()
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                cs3=plt.plot(lon0km,lat0km,ms=10,c='r',marker='+',label='Spill Location')
                plt.legend([cs3],['Spill Location'], loc='lower left',ncol=3, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=coni,label="Field Data",c='b')
                    plt.legend(scatterpoints=1, frameon=False, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                            cs = plt.contour(xaxis,yaxis,fhcont.reshape(len(xaxis),len(yaxis)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,Dep,cmap=plt.get_cmap('hot'))
                            cs = plt.tricontour(yaxisbat,xaxisbat,fh,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newxaxis,newyaxis,pdepth,cmap=plt.get_cmap('hot'))
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,drivcont,cmap=plt.get_cmap('hot'))
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    level=cs2.levels
                    for i in range(len(level)):
                        lab = str(level[i]) + ' m'
                        labels.append(lab)
                    for i in range(len(labels)):
                        cs2.collections[i].set_label(labels[i])
                    plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=4,bbox_to_anchor=(0,1.02,1,0.2))

        elif ourinformation['Method'] == 'Minimum':                
            if ourinformation['Map'] == 'Coordinate':
                if ourinformation['SpillPlace'] == 'Ocean':
                    plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newy,newx,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                plt.colorbar() 
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                cs3=plt.plot(a.lon0,a.lat0,ms=10,c='r',marker='+',label='Spill Location')
                plt.legend([cs3],['Spill Location'], loc='lower left',ncol=4, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['SpillPlace'] == 'Ocean':
                    cs4=plt.contour(Ya,Xa,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])  
                if ourinformation['SpillPlace'] == 'River':
                    cs4=plt.tricontour(newy,newx,scf[i], levels=[float(ourinformation["level"])], colors=['g'])
                level=cs4.levels[0]                
                cs4.collections[0].set_label('Conf. Bound')
                plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=4)     
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.longi,a.lati,s=coni,label="Field Data",c='b')
                    plt.legend(scatterpoints=1, frameon=False, labelspacing=1,ncol=4,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2)) 
                elif ourinformation['Plot'] =="nofield":  
                    pass
                if ourinformation['contour'] =='contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(batt,bat,Dep,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.tricontour(batt,bat,fhcont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newy,newx,pdepth,cmap=plt.get_cmap('hot'), linewidths=1)
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yrivcont,xrivcont,drivcont,cmap=plt.get_cmap('hot'), linewidths=1)
                        #cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                    level=cs2.levels
                    for i in range(len(level)):
                        lab = str(level[i]) + ' m'
                        labels.append(lab)
                    for i in range(len(labels)):
                        cs2.collections[i].set_label(labels[i])
                    plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=4,bbox_to_anchor=(0,1.02,1,0.2))   
                elif ourinformation['contour'] =="nocontour":
                    pass

            elif ourinformation['Map'] == 'km':
                if ourinformation['SpillPlace'] == 'Ocean': 
                    plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newxaxis,newyaxis,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' # make the test bolder                        
                plt.colorbar()  
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'
                cs3=plt.plot(lon0km,lat0km,0,ms=10,c='r',marker='+',label='Spill Location')
                plt.legend([cs3],['Spill Location'], loc='lower left',ncol=3, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['SpillPlace'] == 'Ocean':
                    cs4=plt.contour(xaxis,yaxis,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])  
                if ourinformation['SpillPlace'] == 'River':
                    cs4=plt.tricontour(newxaxis,newyaxis,scf[i], levels=[float(ourinformation["level"])], colors=['g'])
                level=cs4.levels[0]                
                cs4.collections[0].set_label('Conf. Bound')
                plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3)  
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=coni,label="Field Data",c='b')
                    plt.legend(scatterpoints=1, frameon=False, labelspacing=1,ncol=4,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                            cs = plt.contour(xaxis,yaxis,fhcont.reshape(len(xaxis),len(yaxis)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,Dep,cmap=plt.get_cmap('hot'))
                            cs = plt.tricontour(yaxisbat,xaxisbat,fh,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newxaxis,newyaxis,pdepth,cmap=plt.get_cmap('hot'))
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,drivcont,cmap=plt.get_cmap('hot'))
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    level=cs2.levels
                    for i in range(len(level)):
                        lab = str(level[i]) + ' m'
                        labels.append(lab)
                    for i in range(len(labels)):
                        cs2.collections[i].set_label(labels[i])
                    plt.legend(loc='lower left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))


        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        fffilename  = "Results/sunken"+time+"_back.png"
        plt.savefig(fffilename,bbox_inches="tight")
        plt.clf()
        img = cv2.imread(fffilename)
        crop_img = img[55:480,480:580]
        crop_img2 = img[0:50,60:700]        
        # cv2.imshow("image",crop_img)
        fffilename1 = "Results/sunken"+time+"_legend.png"
        cv2.imwrite(fffilename1,crop_img)
        fffilename2 = "Results/sunken"+time+"_outputlegend.png"        
        cv2.imwrite(fffilename2,crop_img2)
        plt.clf()

        #print Result
    for i in range(len(t)):
        if ourinformation['Method'] == 'Best':                     
            if ourinformation['Map'] == 'Coordinate':    
                if ourinformation['SpillPlace'] == 'Ocean':
                    plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newy,newx,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                    ax = plt.gca()
                    ax.set_facecolor('xkcd:royal')
                #plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
                plt.plot(a.lon0,a.lat0,ms=9,c='r',marker='+',label='Spill Location')
                Xticks = plt.xticks()[0]
                print("11111111111",Xticks)
                Yticks = plt.yticks()[0]
                print("222222222222",Yticks)     
                #print 'Xticks', Xticks 
                #print 'Yticks', Yticks  
                for jj in Yticks:
                    for ii in Xticks:
                        if jj >= 0:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])), "N %.3f%s" %(abs(jj),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w')
                        else:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])) , "S %.3f%s" %(abs(jj),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w')
                        if ii >= 0:
                            plt.text(ii -(0.17*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.95*(Yticks[2]-Yticks[1])), "E %.3f o" %abs(ii), fontsize = 10, weight = 'normal', color ='w', rotation = 90)
                        else:
                            plt.text(ii -(0.17*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.95*(Yticks[2]-Yticks[1])), "W %.3f%s" %(abs(ii),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w', rotation = 90)
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.longi,a.lati,s=coni,color='blue')
                elif ourinformation['Plot'] == 'nofield':
                    pass
                if ourinformation['contour'] =='contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(batt,bat,Dep,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.tricontour(batt,bat,fhcont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newy,newx,pdepth,cmap=plt.get_cmap('hot'), linewidths=1)
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yrivcont,xrivcont,drivcont,cmap=plt.get_cmap('hot'), linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                elif ourinformation['contour'] =='nocontour':
                    pass

            if ourinformation['Map'] == 'km': 
                if ourinformation['SpillPlace'] == 'Ocean': 
                    plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newxaxis,newyaxis,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
                plt.plot(lon0km,lat0km,ms=10,c='r',marker='+',label='Spill Location')
                Xticks = plt.xticks()[0]
                print("11111111111",Xticks)
                Yticks = plt.yticks()[0]
                print("222222222222",Yticks)               
                for jj in Yticks:
                    for ii in Xticks:
                        if jj >= 0:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(jj),u" "), fontsize = 10, weight = 'normal', color ='w')
                        else:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])) , "%.1f%s" %(abs(jj),u" "), fontsize = 10, weight = 'normal', color ='w')
                        if ii >= 0:
                            plt.text(ii -(0.001*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.85*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(ii),u" "), fontsize = 10, weight = 'normal', color ='w', rotation = 90) # 0.85 for upper; 0.001 for distance away from the line
                        else:
                            plt.text(ii -(0.001*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.85*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(ii),u" "), fontsize = 10, weight = 'normal', color ='w', rotation = 90)                
                plt.text(0.2,1.8,"%d,%d"%(ourinformation['y_min'],ourinformation['x_min']),fontsize = 10, weight = 'normal', color ='r')
                plt.text(0.2,1.0,"0 km,0 km",fontsize = 10, weight = 'normal', color ='r')         
                plt.plot(0,0,'ro',ms=20)                
                if ourinformation['Plot'] == 'field':
                    plt.scatter(SXkm,SYkm,s=coni,color='blue')#,marker=r'$\clubsuit$')           
                elif ourinformation['Plot'] == 'nofield':
                    pass
                if ourinformation['contour'] =='contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                            cs = plt.contour(xaxis,yaxis,fhcont.reshape(len(xaxis),len(yaxis)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,Dep,cmap=plt.get_cmap('hot'))
                            cs = plt.tricontour(yaxisbat,xaxisbat,fh,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newxaxis,newyaxis,pdepth,cmap=plt.get_cmap('hot'))
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,drivcont,cmap=plt.get_cmap('hot'))
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                elif ourinformation['contour'] =='nocontour':
                    pass

        elif ourinformation['Method'] == 'Minimum':
            if ourinformation['Map'] == 'Coordinate':   
                if ourinformation['SpillPlace'] == 'Ocean':
                    plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                    plt.contour(Ya,Xa,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])  
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newy,newx,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                    plt.tricontour(newy,newx,scf[i], levels=[float(ourinformation["level"])], colors=['g'])
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
                plt.plot(a.lon0,a.lat0,ms=9,c='r',marker='+',label='Spill Location')
                Xticks = plt.xticks()[0]
                print("11111111111",Xticks)
                Yticks = plt.yticks()[0]
                print("222222222222",Yticks)        
                for jj in Yticks:
                    for ii in Xticks:
                        if jj >= 0:
                            plt.text(Xticks[1]+(0.05*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])), "N %.2f%s" %(abs(jj),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w')
                        else:
                            plt.text(Xticks[1]+(0.05*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])) , "S %.2f%s" %(abs(jj),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w')
                        if ii >= 0:
                            plt.text(ii -(0.17*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.95*(Yticks[2]-Yticks[1])), "E %.2f o" %abs(ii), fontsize = 10, weight = 'normal', color ='w', rotation = 90)
                        else:
                            plt.text(ii -(0.17*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.95*(Yticks[2]-Yticks[1])), "W %.2f%s" %(abs(ii),u"\u00B0"), fontsize = 10, weight = 'normal', color ='w', rotation = 90)
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.longi,a.lati,s=coni,color='blue')
                elif ourinformation['Plot'] == 'nofield':
                    print "nofield"
                if ourinformation['contour'] =='contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.contour(Ya,Xa,fhcont.reshape(len(Xa),len(Ya)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(batt,bat,Dep,cmap=plt.get_cmap('hot'), linewidths=1)
                            cs = plt.tricontour(batt,bat,fhcont,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newy,newx,pdepth,cmap=plt.get_cmap('hot'), linewidths=1)
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yrivcont,xrivcont,drivcont,cmap=plt.get_cmap('hot'), linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                elif ourinformation['contour'] =='nocontour':
                    print 'nocontour'

            elif ourinformation['Map'] == 'km': 
                if ourinformation['SpillPlace'] == 'Ocean': 
                    plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)),levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                if ourinformation['SpillPlace'] == 'River':
                    plt.tricontourf(newxaxis,newyaxis,Result[i],levels=np.round(np.linspace(0,1,15),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.contour(xaxis,yaxis,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
                plt.plot(lon0km,lat0km,ms=10,c='r',marker='+',label='Spill Location')
                Xticks = plt.xticks()[0]
                print("11111111111",Xticks)
                Yticks = plt.yticks()[0]
                print("222222222222",Yticks)               
                for jj in Yticks:
                    for ii in Xticks:
                        if jj >= 0:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(jj),u" "), fontsize = 10, weight = 'normal', color ='w')
                        else:
                            plt.text(Xticks[1]+(0.01*(Xticks[2]-Xticks[1])), jj +(0.1*(Yticks[2]-Yticks[1])) , "%.1f%s" %(abs(jj),u" "), fontsize = 10, weight = 'normal', color ='w')
                        if ii >= 0:
                            plt.text(ii -(0.001*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.85*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(ii),u" "), fontsize = 10, weight = 'normal', color ='w', rotation = 90) # 0.85 for upper; 0.001 for distance away from the line
                        else:
                            plt.text(ii -(0.001*(Xticks[2]-Xticks[1])) , Yticks[1]+(0.85*(Yticks[2]-Yticks[1])), "%.1f%s" %(abs(ii),u" "), fontsize = 10, weight = 'normal', color ='w', rotation = 90)                
                plt.text(0.2,1.8,"%d,%d"%(ourinformation['y_min'],ourinformation['x_min']),fontsize = 10, weight = 'normal', color ='r')
                plt.text(0.2,1.0,"0 km,0 km",fontsize = 10, weight = 'normal', color ='r')         
                plt.plot(0,0,'ro',ms=20)                
                if ourinformation['Plot'] == 'field':
                    plt.scatter(SXkm,SYkm,s=coni,color='blue')#,marker=r'$\clubsuit$')
                    print "I am field"             
                elif ourinformation['Plot'] == 'nofield':
                    print "nofield"
                if ourinformation['contour'] =='contour':
                    if ourinformation['SpillPlace'] == 'Ocean':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                            cs = plt.contour(xaxis,yaxis,fhcont.reshape(len(xaxis),len(yaxis)),cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,Dep,cmap=plt.get_cmap('hot'))
                            cs = plt.tricontour(yaxisbat,xaxisbat,fh,cmap=plt.get_cmap('hot'), linewidths=1, linestyles='dashed')
                        #plt.clabel(cs, inline=0.5,fontsize=8,linewidths=1)
                        plt.clabel(cs2, inline=0.5,fontsize=8,linewidths=1)
                    if ourinformation['SpillPlace'] == 'River':
                        if ourinformation['SunkenUpload'] == 'No Upload':
                            cs2 = plt.tricontour(newxaxis,newyaxis,pdepth,cmap=plt.get_cmap('hot'))
                        if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
                            cs2 = plt.tricontour(yaxisbat,xaxisbat,drivcont,cmap=plt.get_cmap('hot'))
                        plt.clabel(cs2, inline=0.5,fontsize=8)
                elif ourinformation['contour'] =='nocontour':
                    print 'nocontour'           

        #pyplot.text(self.lon0-0.005, self.lat0-0.005, "x", fontsize = 16, weight = 'bold', color ='m')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        filename  = "Results/sunken"+time+".png"
        # plt.show()
        plt.savefig(filename, dpi=599, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False)
        figPGW = np.array([(float(ourinformation['x_max']) - float(ourinformation['x_min']))/4792.0, 0.0, 0.0, -(float(ourinformation['y_max']) - float(ourinformation['y_min']))/3600.0, float(ourinformation['x_min']), float(ourinformation['y_max'])])
        figPGW = np.array([(float(Xticks[len(Xticks)-1]) - float(Xticks[0]))/4792.0, 0.0, 0.0, -(float(Yticks[len(Yticks)-1]) - float(Yticks[0]))/3600.0, float(Xticks[0]), float(Yticks[len(Yticks)-1])])
        filename11 = "Results/sunken"+time+".pgw"
        figPGW.tofile(filename11 , sep = '\n', format = "%.16f")

        img = cv2.imread(filename,1)
        # SaltImage=move(img)
        num_rows, num_cols = img.shape[:2]
        if ourinformation['SpillPlace'] == 'Ocean':
            if ourinformation['Map'] == 'Coordinate':
                translation_matrix = np.float32([ [1,0,560], [0,1,480] ])
                img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
                img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 635, num_rows + 240))
            if ourinformation['Map'] == 'km':
                translation_matrix = np.float32([ [1,0,0], [0,1,0] ])
                img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
                img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
            
        if ourinformation['SpillPlace'] == 'River':
            #translation_matrix = np.float32([ [1,0,640], [0,1,320] ]) #0.4, 0.4 scale
            translation_matrix = np.float32([ [1,0,490], [0,1,350] ]) #0.15,0.05 scale
            #translation_matrix = np.float32([ [1,0,300], [0,1,480] ]) #0.2,0.1 scale
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 635, num_rows + 240))
            # translation_matrix = np.float32([ [1,0,500], [0,1,340] ])
            # img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
            # img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 635, num_rows + 240))            
        img_translation[np.where((img_translation == [0,0,0]).all(axis = 2))] = [135,0,0]        
        cv2.imwrite("Results/sunken"+time+".png",img_translation)
        filename1 = "Results/sunken"+time+".png" 
        return filename1,fffilename1,fffilename2

    # plt.show()

