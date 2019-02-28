# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from math import *
#import numpy as numpy
import random
#import mcint
import utm
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from functools import partial
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import integrate
#import vegas
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
#import pp 
import math, sys, time 
import scipy.optimize as optimize
from  scipy.optimize import differential_evolution
import cv2
import geopy.distance

ourinformation = {}
parameter = []

def B_sampling(DLx,DLy,mux,muy,sigmax,sigmay,ro): #Q2
    """Definition of term B in the Bivariate normal Gaussian Distribution using the sampling points"""
    Bs=((((DLx-mux))**(2.0))/((sigmax)**2.0))+((((DLy-muy))**(2.0))/((sigmay)**2.0))-((2.0*(ro)*(DLx-mux)*(DLy-muy))/(sigmax*sigmay))
    return Bs

def CG(sigmax,sigmay,BuSamp,ro):
    """Conditional Gaussian function:"""
    CG=((1.0)/(2.0*(np.pi)*sigmax*sigmay*(np.sqrt(1-((ro)**2.0)))))*(np.exp(-(BuSamp)/(2.0*(1.0-((ro)**2.0)))))
    return CG

# ff defines the gaussian function
def ff(x,y,vx,vy,Dx,Dy,ro,x0,y0,Dx0,Dy0,t,s):
    k=0
    for i in range(len(s)):
        mux = x0 + vx*(t-s[i])
        muy = y0 + vy*(t-s[i])
        sigmax = np.sqrt(2.0*Dx*(t-s[i]))+Dx0
        sigmay = np.sqrt(2.0*Dy*(t-s[i]))+Dy0
        mm = CG(sigmax,sigmay,B_sampling(x,y,mux,muy,sigmax,sigmay,ro),ro)
        k = k + mm/len(s)#1/len(s)*ff(x,y,mux,muy,sx,sy,ro[i])
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
    # delta_vx = 0.2
    # delta_vy = 0.2
    Dx_min = float(ourinformation['dxmin'])  # user's inputs Dx_min
    Dx_max = float(ourinformation['dxmax'])   # user's inputs Dx_max
    # delta_Dx = 5
    Dy_min = float(ourinformation['dymin'])  # user's inputs Dy_min
    Dy_max = float(ourinformation['dymax'])   # user's inputs Dy_min

    # delta_Dy = 5
    ro_min = -0.999
    ro_max = 0.999
    # delta_ro = 0.4
    gamma_min = 0.00
    gamma_max = 1.0

    random.seed(N)
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

    np.random.seed(N)
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

# def IniLikelihood(a,parameter):
def IniLikelihood(a,parameter):
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    DLx=a.DLx
    DLy=a.DLy
    x0=a.x0
    y0=a.y0
    DLcon=a.DLcon
    Dx0=a.Dx0
    Dy0=a.Dy0
    st=a.st 
    ss1 = a.ss1
    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0
    Prob = 0
    CompLikelihood=1
    
    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss1[ci]) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss1[ci]) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss1[ci]) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob                
                    IniIndLikelihood[ci] = np.log(Lamda)-(Lamda*DLcon[ci])#(Lamda*np.exp(-Lamda*DLcon[ci]))/(1-np.exp(-100*Lamda))#(np.log(Lamda)-(Lamda*DLcon[ci]))-(100*Lamda)##(np.log(Lamda))-(Lamda*DLcon[ci])-(np.log(1-np.exp(-100*Lamda)))
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
        vx1,vx2,vx3,vx4=parameter
        #vx2=parameter[1]
        #vx3=parameter[2]
        #vx4=parameter[3]
        Max = a.MaxLogLikeP
        #print Max

        r = a.newr
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

        #DLx=a.DLx
        #xx=a.xx 
        DLx = np.concatenate((a.DLx,a.xx),axis=None)

        #DLy=a.DLy
        #yy=a.yy 
        DLy = np.concatenate((a.DLy,a.yy),axis=None)
        x0=a.x0
        y0=a.y0
        #DLcon=a.DLcon
        #Cprior=a.Cprior
        DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
        Dx0=a.Dx0
        Dy0=a.Dy0
        #st= a.st
        ptt = np.array([a.pt]*len(a.xx))
        st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
        ss3=[]
        for i in range(len(ptt)):
            K=1
            s3=np.zeros(K)
            ss3.append(s3)
        #ss3=np.array(ss3)
        s1=a.ss1+ss3
        #ss1=np.concatenate((a.ss1,ss3),axis=0)

        IniIndLikelihood = np.ones([len(DLx)])
        Lamda = 0# np.zeros([len(vx1)])
        Prob = 0#np.zeros([len(vx1)])
        CompLikelihood=1##np.ones([len(vx1)])
        # print len(x),range(31)

        for ci in range(len(DLx)):
            if DLcon[ci] >0:
                for i in range(1):
                    ss=s1[ci]
                    ss=ss[ss<st[ci]]
                    Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                    +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                    +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                    +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)
                    if Prob>1e-308:
                        Lamda = 1/Prob                    
                        IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                    else:
                        Lamda = 0
                        IniIndLikelihood[ci] = 0  
                    
        for i in range(1):
            for ci in range(len(DLx)):
                if DLcon[ci]>0:
                    if IniIndLikelihood[ci] == 0:
                        CompLikelihood = 0
        d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
        login = abs(np.sum(IniIndLikelihood) - Max + d/2)
        #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
        #print login 
        return login 

def IniLikelihood2(a,parameter):
    #gamma1,gamma2,gamma3,gamma4,ro1,ro2,ro3,ro4,Dx4,Dy1,Dy2= parameter
    vy1,vy2,vy3,vy4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    Max = a.MaxLogLikeP
    r = a.newr
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

    #DLx=a.DLx
    #xx=a.xx 
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    #ss1=a.ss1 
    s1=a.ss1+ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0  
                    
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0
    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
    #print login 
    return login 

def IniLikelihood3(a,parameter):
    Dx1,Dx2,Dx3,Dx4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    Max = a.MaxLogLikeP
    r = a.newr    
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

    #DLx=a.DLx
    #xx=a.xx 
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    #ss1=a.ss1 
    s1=a.ss1+ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0  
                    
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0
    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
    #print login
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
    r = a.newr
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19] 
    #DLx=a.DLx
    #xx=a.xx 
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    #ss1=a.ss1 
    s1=a.ss1+ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0  
                    
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0
    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
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
    r = a.newr
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    #DLx=a.DLx
    #xx=a.xx 
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    #ss1=a.ss1 
    s1=a.ss1+ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0  
                    
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0
    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
    return login 

def IniLikelihood6(a,parameter):
    gamma1,gamma2,gamma3,gamma4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    params3 = a.fitted_params3
    Dx1,Dx2,Dx3,Dx4 = params3
    params4 = a.fitted_params4
    Dy1,Dy2,Dy3,Dy4 = params4
    params5 = a.fitted_params5
    ro1,ro2,ro3,ro4 = params5
    Max = a.MaxLogLikeP
    #DLx=a.DLx
    #xx=a.xx 
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        K=len(a.ss1[0])
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    #ss1=a.ss1 
    s1=a.ss1+ss3

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                Prob = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],s1[ci]) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],s1[ci]) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],s1[ci]) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],s1[ci])
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#/(1-np.exp(-100*Lamda))
                else:
                    Lamda = 0
                    IniIndLikelihood[ci] = 0  
                    
    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0
    d = chi2.ppf(1-float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #login = (np.sum(IniIndLikelihood) - Max + d/2)**2
    return login         
            
#_______________________________regular likelihoods_________________________________________________

def Likelihood(a,N):
    global parameter
    parameter=sampler(N)
    IniLikelihood=np.array(multicore1(a,parameter))

    IniIndLikelihood = np.transpose(IniLikelihood)
    DLcon = a.DLcon
    DLx = a.DLx 
    DLy = a.DLy
    CompLikelihood =np.ones(N)
    Likelihood = np.zeros(N)
    Likelihoodi = np.zeros(N)
    #Likelihoodi = np.zeros(N)
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
            #Likelihood[i] = Likelihood[i]/np.max(Likelihood[i])
        if CompLikelihood[i]==1:
            if MaxLogLike ==-22:
                MaxLogLike=Likelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,Likelihood[i]])
    print MaxLogLike

    for i in range(N):
        if CompLikelihood[i]==1:
            Likelihood[i] = Likelihood[i] - MaxLogLike #+350 
    #print Likelihood #+10000
            Likelihoodi[i] = np.exp(Likelihood[i])

    return Likelihoodi

def IniLikelihoodP(a,parameter):
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    #DLx=a.DLx
    #xx=a.xx
    DLx = np.concatenate((a.DLx,a.xx),axis=None)

    #DLy=a.DLy
    #yy=a.yy 
    DLy = np.concatenate((a.DLy,a.yy),axis=None)
    x0=a.x0
    y0=a.y0
    #DLcon=a.DLcon
    #Cprior=a.Cprior#*(10**-10)
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    Dx0=a.Dx0
    Dy0=a.Dy0
    #st= a.st
    ptt = np.array([a.pt]*len(a.xx))
    st = np.concatenate((a.st,np.array([a.pt]*len(a.xx))),axis=None)
    ss3=[]
    for i in range(len(ptt)):
        #K=len(a.ss1[0])
        K=1
        s3=np.zeros(K)
        ss3.append(s3)
    #ss3=np.array(ss3)
    s1 = a.ss1+ss3


    IniIndLikelihoodP = np.ones([len(DLx)])
    LamdaP = 0# np.zeros([len(vx1)])
    ProbP = 0#np.zeros([len(vx1)])
    CompLikelihoodP=1##np.ones([len(vx1)])
    # print len(x),range(31)

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s1[ci]
                ss=ss[ss<st[ci]]
                ProbP = gamma1*ff(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma2*ff(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma3*ff(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,st[ci],ss) \
                 +gamma4*ff(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,st[ci],ss)

                if ProbP>1e-308:
                    LamdaP = 1/ProbP                 
                    IniIndLikelihoodP[ci] = np.log(LamdaP)-LamdaP*DLcon[ci]#(LamdaP*np.exp(-LamdaP*Cprior[ci]))/(1-np.exp(-100*LamdaP))#(np.log(LamdaP)-(LamdaP*Cprior[ci]))-(100*LamdaP)##(np.log(LamdaP))-(LamdaP*Cprior[ci])-(np.log(1-np.exp(-100*LamdaP)))
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
    parameter = a.r
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0
    #x=a.x
    #y=a.y
    [x,y]=loc
    t=a.t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    ss2 = a.ss2

    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,t,ss2)

    return ProObsGivenPar

def LikelihoodNew(a,N):
    global parameter
    parameter=sampler(N)
    IniLikelihoodP=np.array(multicore3(a,parameter))
    IniIndLikelihoodP = np.transpose(IniLikelihoodP)
    #print IniIndLikelihoodP
    Depth = a.Depth
    #xx = a.xx 
    #DLx=a.DLx
    DLx = np.concatenate((a.DLx,a.xx),axis=None)
    #Cprior = a.Cprior#*(10**-10)
    #DLcon = a.DLcon
    DLcon = np.concatenate((a.DLcon,a.Cprior),axis=None)
    CompLikelihoodP =np.ones(N)
    LikelihoodP = np.zeros(N)
    LikelihoodPi = np.zeros(N)
    #LikelihoodPconf = numpy.zeros(N)
    #newLikelihood = numpy.zeros(N)
    #newLikelihoodconf = numpy.zeros(N)
    if np.min(Depth) == np.max(Depth):
        for i in range(N):
            LikelihoodP[i] = 1.0
    else:
        for i in range(N):                                                                                                                                                                                                                                  
            for ci in range(len(DLx)):
                if DLcon[ci]>0:
                    if IniIndLikelihoodP[ci,i] == 0:
                        CompLikelihoodP[i] = 0
        MaxLogLikeP=-22
        for i in range(N):
            for ci in range(len(DLx)):
                if DLcon[ci]>0:
                    if CompLikelihoodP[i]==1:
                        LikelihoodP[i] =  LikelihoodP[i] + IniIndLikelihoodP[ci,i]
                        
            if CompLikelihoodP[i]==1:
                if MaxLogLikeP ==-22:
                    MaxLogLikeP=LikelihoodP[i]
                else:
                    MaxLogLikeP=np.max([MaxLogLikeP,LikelihoodP[i]])
        print MaxLogLikeP

        for i in range(N):
            if CompLikelihoodP[i]==1:
                #LikelihoodPconf[i] = LikelihoodP[i]
                LikelihoodP[i] = LikelihoodP[i] - MaxLogLikeP #+700
                LikelihoodPi[i] = np.exp(LikelihoodP[i])
                #LikelihoodP[i] = LikelihoodP[i]/np.sum(LikelihoodP[i])
        #print LikelihoodP

    # for i in range(N):
    #   newLikelihoodconf[i] = Likelihoodconf[i]+LikelihoodPconf[i]
    #   newLikelihood[i] = Likelihood[i]+LikelihoodP[i] #- 700
    #   newLikelihood[i] = numpy.exp(newLikelihood[i])
    return MaxLogLikeP, LikelihoodPi

def newinteg(a,loc):
    global parameter
    parameter = a.newr
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0

    [x,y]=loc
    t=a.t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    ss2 = a.ss2
    #print ss2

    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,t,ss2) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,t,ss2)

    return ProObsGivenPar

def integcf(a,loc):
    global parameter
    parameter = a.par
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0
    [x,y]=loc
    t=a.t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    s = a.ss2
    ProObsGivenPar = gamma1*ff(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,Dx0,Dy0,t,s) \
    +gamma2*ff(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,Dx0,Dy0,t,s) \
    +gamma3*ff(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,Dx0,Dy0,t,s) \
    +gamma4*ff(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,Dx0,Dy0,t,s)
    return ProObsGivenPar


def multicore1(a,parameter):
        pool = mp.Pool(19)
        res = pool.map(partial(IniLikelihood,a),parameter)
        return res

# def multicore1(a,parameter):
#     ppservers = ()
#     res = []
#     if len(sys.argv) > 1:
#         ncpus = int(sys.argv[1])
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#     else:
#         job_server = pp.Server(ppservers=ppservers)
#     print "Starting pp with", job_server.get_ncpus(), "workers"

#     jobs = [job_server.submit(IniLikelihood,(a,input),(B_sampling,CG,ff),("numpy",)) for input in parameter]
#     for job in jobs:
#         res.append(job())
#     return res

def multicore2(a,loc):
        pool = mp.Pool(19)
        res = pool.map(partial(integ,a),loc)
        return res

# def multicore2(a,loc):
#     ppservers = ()
#     res = []
#     if len(sys.argv) > 1:
#         ncpus = int(sys.argv[1])
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#     else:
#         job_server = pp.Server(ppservers=ppservers)
#     print "Starting pp with", job_server.get_ncpus(), "workers"

#     jobs = [job_server.submit(integ,(a,input),(B_sampling,CG,ff),("numpy",)) for input in loc]
#     for job in jobs:
#         res.append(job())
#     return res

def multicore3(a,parameter):
        pool = mp.Pool(19)
        res = pool.map(partial(IniLikelihoodP,a),parameter)
        return res

# def multicore3(a,parameter):
#     ppservers = ()
#     res = []
#     if len(sys.argv) > 1:
#         ncpus = int(sys.argv[1])
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#     else:
#         job_server = pp.Server(ppservers=ppservers)
#     print "Starting pp with", job_server.get_ncpus(), "workers"

#     jobs = [job_server.submit(IniLikelihoodP,(a,input),(B_sampling,CG,ff),("numpy",)) for input in parameter]
#     for job in jobs:
#         res.append(job())
#     return res

def multicore4(a,loc):
        pool = mp.Pool(19)
        res = pool.map(partial(newinteg,a),loc)
        return res
# def multicore4(a,parameter):
#         pool = mp.Pool(4)
#         res = pool.map(partial(integP,a),parameter)
#         return res

def multicore5(a,loc):
        pool = mp.Pool(19)
        res = pool.map(partial(integcf,a),loc)
        return res

# def multicore4(a,loc):
#     ppservers = ()
#     res = []
#     if len(sys.argv) > 1:
#         ncpus = int(sys.argv[1])
#         job_server = pp.Server(ncpus, ppservers=ppservers)
#     else:
#         job_server = pp.Server(ppservers=ppservers)
#     print "Starting pp with", job_server.get_ncpus(), "workers"

#     jobs = [job_server.submit(newinteg,(a,input),(B_sampling,CG,ff),("numpy",)) for input in loc]
#     for job in jobs:
#         res.append(job())
#     return res

def extract_key(v):
        return v[0]

class Preliminars: # SOSim
    def __init__(self): 
        self.w = 4
        self.u = self.w + 1

class soscore(Preliminars):
    # def __init__(self,InputFileName):
    def __init__(self,datalist):
        Preliminars.__init__(self)
        #Input data
        # datalist = pd.read_csv(InputFileName)
        # lat = datalist['lat']
        # lon = datalist['lon']
        # SpillT = datalist['SpillTime']
        # PredictT = datalist['PredictTime']
        # Scale = datalist['Scale']
        # Node = datalist['Node']
        # OilType = datalist['OilType']
        # SpillPlace = datalist['SpillPlace']
        # Bathymetry = datalist['BathymetryUpload'] #the user will click 'Bathymetry Upload' or 'No Upload'
        # UploadType = datalist['UploadType'] #this is if the user clicks 'UTM coord' or 'Decimal degrees'
        # lat = np.array(lat[~np.isnan(lat)])
        # lon = np.array(lon[~np.isnan(lon)])
        # SpillT = SpillT[~pd.isnull(SpillT)]
        # PredictT = PredictT[~pd.isnull(PredictT)]
        # Scale = np.array(Scale[~np.isnan(Scale)])
        # Node = np.array(Node[~np.isnan(Node)])
        # OilType = int(OilType[~np.isnan(OilType)])
        # SpillPlace = str(SpillPlace[0])
        # BathyUpload = str(Bathymetry[0])
        # UploadType = str(UploadType[0])

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
        # SampleT = SampleT[~pd.isnull(SampleT)]
        PredictT = np.array(PredictT)[~pd.isnull(PredictT)]
        Scale = np.array(Scale)[~np.isnan(Scale)]
        Node = np.array(Node)[~np.isnan(Node)]
        OilType = np.array(OilType)[~np.isnan(OilType)]

        sigmax0 = 0.050
        sigmay0 = 0.050
        #define spill point
        coord0 = utm.from_latlon(lat, lon)
        x0 = coord0[0]/1000.0
        y0 = coord0[1]/1000.0
        #x0=np.pad(x0, (0,3), 'constant')
        #y0=np.pad(y0, (0,3), 'constant')
        duration = [CalTime(SpillT[0],SpillT[1])]
        dura = np.pad(duration,(0,1),'constant')
        t = [CalTime(SpillT[0],PredictT[vld]) for vld in range(len(PredictT))]
        #t = np.array(t)

        ss2 = []

        for i in range(len(t)):
            after = CalTime(PredictT[i],SpillT[1])
            before = CalTime(SpillT[0],PredictT[i])
            if dura[0] != dura[1] and after > 0:
                K = ceil(before)
                s2 = np.linspace(0.0,before,K)
                ss2.append(s2)
            elif SpillT[0] < SpillT[1]:
                K = ceil(dura[0])+1
                s2 = np.linspace(dura[1],dura[0],K)
                ss2.append(s2)
            else:
                K = 1
                s2 = np.zeros(K)
                ss2.append(s2)

        ss2 = np.array(ss2)
        #print ss2

        self.x0 = x0
        self.y0 = y0
        self.SpillT = SpillT
        self.PredictT = PredictT
        self.Scale = Scale
        # self.Node = Node
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

        #Load campaign data
    def UploadCampaign(self,CampaignFileName):
        DLx = []
        DLy = []
        DLcon = []
        st = [] 
        dx = []
        dy = []
        dcon = []
        ST = []# sample time
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

        # for i in range(len(CampaignFileName)):
            # campdata= pd.read_csv(CampaignFileName[i])

        SampleT = campdata['SampleTime']
        SampleT = SampleT[~pd.isnull(SampleT)]
        DLlat = np.array(campdata["lat"])
        DLlon = np.array(campdata["Lon"])
        DLc = np.array(campdata["Con"])
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
        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(lati,longi)])
        DLx.append(np.array(map(float,camdatalist[:,0]))/1000)

        DLy.append(np.array(map(float,camdatalist[:,1]))/1000)
#            
             
        for s in DLcl:
            if s == 0.0:
                conValue = (0.01)
            else:
                conValue = (s/100.0)
            DLcon.append(conValue)
            

        ss1 = []

        for i in range(len(tt)):
            after = CalTime(SampleT[i],self.SpillT[1])
            before = CalTime(self.SpillT[0],SampleT[i])

            if self.dura[0] != self.dura[1] and after > 0:
                K = ceil(before)
                #s1 = numpy.linspace(0.0,ST[i]-1,K)
                s1 = np.linspace(0.0,before,K)
                ss1.append(s1)
            elif self.SpillT[0] < self.SpillT[1]:
                K = ceil(self.dura[0])+1
                s1 = np.linspace(self.dura[1],self.dura[0],K)
                ss1.append(s1)
            else:
                #K = int(before)
                K = 1 
                s1 = np.zeros(K)
                ss1.append(s1)
        print ss1

        #ss1 = numpy.array(ss1)
        #print ss1

        DLx[0] = np.array(DLx[0])
        DLy[0] = np.array(DLy[0])   
        self.DLx = DLx[0]
        self.DLy = DLy[0]
        self.DLcon = DLcon
        self.st = tt
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
        x = []
        y = []
        BathymetryFileName = []
        BathymetryFileName.append(BathymetryFile)

        for i in range(len(BathymetryFileName)):
            bathdata= pd.read_csv(BathymetryFileName[i])
            
            # if 'UTM coord' is clicked
            # if self.UploadType == 'UTM coord':
            if ourinformation['SunkenUpload'] == 'UTM coord':
                print "UTM coord" 
                dat = np.array(bathdata)
                x = dat[:,0]/1000
                y = dat[:,1]/1000
                d = dat[:,2]

            # if 'Decimal degrees' is clicked
            # if self.UploadType == 'Decimal degrees':
            if ourinformation['SunkenUpload'] == 'Decimal degrees':
                print "Decimal degrees"
                dat = np.array(bathdata)
                xdd = dat[:,1]
                ydd = dat[:,0]
                d = dat[:,2]
                camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(xdd,ydd)])
                x.append(np.array(map(float,camdatalist[:,0]))/1000.)
                y.append(np.array(map(float,camdatalist[:,1]))/1000.)
                x = x[0]
                y = y[0]

        xlow = self.lat0 - self.scale[0]
        xhigh = self.lat0 + self.scale[0]
        ylow = self.lon0 - self.scale[1]
        yhigh = self.lon0 + self.scale[1]
        coordl = np.array([utm.from_latlon(xlow,ylow)])
        coordh = np.array([utm.from_latlon(xhigh,yhigh)])
        xlow = float(coordl[0][0])/1000.
        ylow = float(coordl[0][1])/1000.
        xhigh = float(coordh[0][0])/1000. 
        yhigh = float(coordh[0][1])/1000.
        xl = zip(x,y,d)
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
            retardation = 7.0
        if self.OilType == 2.0:
            retardation = 5.6
        if self.OilType == 3.0:
            retardation = 4.2
        if self.OilType == 4.0:
            retardation = 2.8
        if self.OilType == 5.0:
            retardation = 1.4
        if self.OilType == 6.0:
            retardation = 0.0
        self.retardation = retardation
        #self.st = self.st - retardation
        self.pt = self.st[0]
        #self.pt = 0.0
        K = 1
        self.ss3 = np.zeros(K)
        #self.ss3 = 0.0
        B = [max(self.DLcon)]
        #print B
        hiIndex = B.index(max(B))
        latestST = max(self.st)
        #self.t = np.array(self.t) - retardation # time to the maximum concentration campaign


    def x0y0DueSinkingRetardation(self):
        B = np.max(self.DLcon)
        C = np.argmax(self.DLcon)     
    
        x0news = self.DLx[C]
        y0news = self.DLy[C]
        
        x0new = x0news
        y0new = y0news

        x0 = self.x0
        y0 = self.y0
        oilType = self.OilType

        distX = np.array(x0new - x0)
        distY = np.array(y0new - y0)
        if oilType == 1.0:
            sunkx0 = (x0 + (7.0*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (7.0*(np.array(distY)/8.0)))#*B
        if oilType == 2.0:
            sunkx0 = (x0 + (5.6*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (5.6*(np.array(distY)/8.0)))#*B
        if oilType == 3.0:
            sunkx0 = (x0 + (4.2*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (4.2*(np.array(distY)/8.0)))#*B
        if oilType == 4.0:
            sunkx0 = (x0 + (2.8*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (2.8*(np.array(distY)/8.0)))#*B
        if oilType == 5.0:
            sunkx0 = (x0 + (1.4*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (1.4*(np.array(distY)/8.0)))#*B
        if oilType == 6.0:
            sunkx0 = (x0 + (0.0*(np.array(distX)/8.0)))#*B
            sunky0 = (y0 + (0.0*(np.array(distY)/8.0)))#*B
        self.sunkx0 = sunkx0
        self.sunky0 = sunky0


# if __name__ == "__main__":

def sunken_main(myinformation,progressBar):
    #delta_gamma = 0.1

    print myinformation

    global ourinformation

    ourinformation = myinformation
    a = soscore(ourinformation)
    a.ourinformation = ourinformation
    progressBar.setValue(15)


    # a = soscore("datainputA.csv")
    # a.UploadCampaign(["athos1.csv"])

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
    BathyUpload = a.BathyUpload


    DLlat = a.DLlat 
    DLlon = a.DLlon

    DLcon = a.DLcon

    XSpill = a.sunkx0
    YSpill = a.sunky0


    # Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.Node+1)
    # Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.Node+1)
    Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.xNode+1)
    Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.yNode+1)

    coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(Xa,Ya)])
    xa = np.array(map(float,coord[:,0]))/1000
    ya = np.array(map(float,coord[:,1]))/1000
    [x,y] = np.meshgrid(xa,ya)
    x = np.concatenate(x)
    y = np.concatenate(y)

    gd = ceil(np.sqrt(len(DLcon)))
    n = int(gd) - 1

    X=SX
    Y=SY
    t = a.t
    print t
    SP = a.st
    pt = a.pt
    ss1 = a.ss1
    ss2 = a.ss2
    ss3 = a.ss3
    Dx0 = 0.05
    Dy0 = 0.05

    a.DLcon=a.DLcon
    a.DLx=a.DLx
    a.DLy=a.DLy
    a.x0=a.sunkx0
    a.y0=a.sunky0
 
    N=10000
    # N = 100


    prob = Likelihood(a,N)
    a.prob = prob
    global parameter
    parameter = sampler(N)
    print parameter[0]

    Likemle = np.argmax(prob)
    a.r = parameter[Likemle]
    print a.r

    rr = np.array(parameter[Likemle])
    gammamle = rr[20:24]
    gmax = np.argmax(gammamle)
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
    Conmax = np.argmax(DLcon)
    xCmax = SX[Conmax]
    yCmax = SY[Conmax]
    kx = (xCmax - (x0 + vxm*SP[0]))/(Dx0 + np.sqrt(2.0*Dxm*SP[0]))
    ky = (yCmax - (y0 + vym*SP[0]))/(Dy0 + np.sqrt(2.0*Dym*SP[0]))
    Pmaxnorm = np.max(DLcon)/(np.exp(((kx**2.0)+(ky**2.0)-(2.0*rom*kx*ky))/(2.0*(1.0-(rom**2.0)))))
    print Pmaxnorm

    a.x=x
    a.y=y
    t=a.t
    res = []
    for u in range(len(t)):
        a.t = t[u]
        a.ss2 = ss2[u]
        print a.t,a.ss2
        resa = []
        loc = zip(x,y)
        resa=np.array(multicore2(a,loc))
        resa=np.transpose(resa)
        sum = 0
        #for i in resa:
        #   sum = sum + np.array(i)
        res.append(resa)
    s = np.array(res)

    #This part is if the User clicks the 'No Upload' button.
    # if BathyUpload == 'No Upload': 
    if ourinformation['SunkenUpload'] == 'No Upload':
        print "No Upload"
        xoc = Xa[0::len(Xa)//n-1]
        yoc = Ya[0::len(Ya)//n-1]
        px = []
        py = []
        for i in range(0,6):
            px.append(random.uniform(xoc[i],xoc[i+1]))
            py.append(random.uniform(yoc[i],yoc[i+1]))
        #print px,py
        db = oceansdb.ETOPO()
        d = db['topography'].extract(lat=px, lon=py)
        de = d['height']
        #print de
        #dep = numpy.concatenate(de, axis=0)
        for i in range(len(de)):
            for j in range(len(de[i])):
                if de[i][j] >= 0.0:
                    de[i][j] = 0.0
        #print de
        Dep = abs(de)
        print Dep 

        coordp = np.array([utm.from_latlon(i,j) for (i,j) in zip(px,py)])
        xp = np.array(map(float,coordp[:,0]))/1000
        yp = np.array(map(float,coordp[:,1]))/1000
        #print xp
        [yyy,xxx] = np.meshgrid(yp,xp)
        loc = []
        for i in range(len(de)):
            loc.append(zip(xxx[i],yyy[i],Dep[i]))
        #print loc
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
        xx = newxx
        yy = newyy 
        Depth = newd
        #xp = xa[0::len(xa)//n]
        #print xp
        #yp = ya[0::len(ya)//n]
        
        #xx = numpy.concatenate(xx)
        #yy = numpy.concatenate(yy)
        #print xx
        a.x=x
        a.y=y
        t=[a.t]
        #t=np.array(a.t)
        print t,ss2
        #t=np.array(t)
        resR = []
        for u in range(len(t)):
            a.t = t[u]
            a.ss2 = ss2[u]
            resaR = []
            loc = zip(xx,yy)
            resaR=np.array(multicore2(a,loc))
            resaR=np.transpose(resaR)
            sum = 0
            #for i in resaR:
            #    sum = sum + np.array(i)
            resR.append(resaR)
        sR = np.array(resR)
        ResultP = sR[0]

     #This part is if the User clicks the 'Bathymetry Upload' button.
    # if BathyUpload == 'Bathymetry Upload':
    if ourinformation['SunkenUpload'] == 'UTM coord' or ourinformation['SunkenUpload'] == 'Decimal degrees':
        #this function 'UploadBathymetry' is used if the 'Bathymetry Upload' button is clicked.  The user clicked 'UTM coord' or 'Decimal degrees' and one of these buttons is connected to this function.
        
        # a.UploadBathymetry(["delriver.csv"])
        print "Bathymetry Upload"
        a.UploadBathymetry(ourinformation['HydroButton'])

        bat = a.bat 
        batt = a.batt 
        Dep = a.Dep

        xp = bat[0::len(bat)//n]
        yp = batt[0::len(batt)//n]
        [xx,yy] = np.meshgrid(xp,yp)
        xx = np.concatenate(xx)
        yy = np.concatenate(yy)
        depth = []
        coordinate = zip(xx,yy,Dep)
        for i in range(len(coordinate)):
            for j in range(len(xx)):
                if coordinate[i][0] == xx[j] and coordinate[i][1] == yy[j]:
                    depth.append(coordinate[i][2])
        depth = np.array(depth)
        for i in range(len(depth)):
            if depth[i] >= 0.0:
                depth[i] = 0.0
        Depth = abs(depth)
        print Depth 
        a.x=x
        a.y=y
        t=[a.t]
        #t=np.array(a.t)
        print t,ss2
        #t=np.array(t)
        resR = []
        for u in range(len(t)):
            a.t = t[u]
            a.ss2 = ss2[u]
            resaR = []
            loc = zip(xx,yy)
            resaR=np.array(multicore2(a,loc))
            resaR=np.transpose(resaR)
            sum = 0
            #for i in resaR:
            #    sum = sum + np.array(i)
            resR.append(resaR)
        sR = np.array(resR)
        ResultP = sR[0]

    

    Cprior = (np.sqrt((((Depth-np.min(Depth))/(np.max(Depth)-np.min(Depth)))*(ResultP/np.max(ResultP)))))*Pmaxnorm
    print Cprior

    a.xx=xx
    a.yy=yy
    a.pt=pt
    a.Cprior=Cprior
    a.Cprior=a.Cprior
    a.Depth = Depth 


    maxlog = LikelihoodNew(a,N)[0] #+ np.log(LikelihoodNew(a,N)[0][np.argmax(LikelihoodNew(a,N)[0])])

    probNew = LikelihoodNew(a,N)[1]
    newLikemle = np.argmax(probNew)
    #print probNew
    #print newLikemle
    logmax = np.log(probNew[newLikemle])
    print logmax
    #print np.max(probNew)
    a.MaxLogLikeP = maxlog #+ logmax
    a.newr = parameter[newLikemle]
    print a.newr
    progressBar.setValue(50)

#____________________________confidence bounds_________________________________________

    if ourinformation['Method'] == 'Minimum':  
        bounds=[(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax']))] 
        #bounds = [a.newr[0]]   
        result = differential_evolution(partial(IniLikelihood1,a),bounds)
        print result

        if result.success:
            fitted_params = result.x
            print(fitted_params)
        else:
            raise ValueError(result.message)
        #a.fitted_params = fitted_params.tolist()
        
        vx1 = fitted_params[0]
        vx2 = fitted_params[1]
        vx3 = fitted_params[2]
        vx4 = fitted_params[3]

        a.fitted_params = vx1,vx2,vx3,vx4
        print a.fitted_params#,Dx1#,Dx2#,Dx3

        bounds=[(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vymin']),float(ourinformation['vymax']))]    
        result2 = differential_evolution(partial(IniLikelihood2,a),bounds) 
        print result2
        #initial_guess = [a.r[4], a.r[5], a.r[6], a.r[7]]
        #result2 = optimize.minimize(partial(IniLikelihood2,a), initial_guess,method = 'SLSQP',bounds=[(-3.0,3.0),(-3.0,3.0),(-3.0,3.0),(-3.0,3.0)])#method = 'TNC'method='nelder-mead') #method='Powell')method='nelder-mead')#method = 'SLSQP')#method='nelder-mead')#method='COBYLA')#method='nelder-mead')
        if result2.success:
            fitted_params2 = result2.x
            print(fitted_params2)
        else:
            raise ValueError(result2.message)    
        #a.fitted_params2 = fitted_params2.tolist()
        vy1 = fitted_params2[0]
        vy2 = fitted_params2[1]
        vy3 = fitted_params2[2]
        vy4 = fitted_params2[3]

        a.fitted_params2 = vy1,vy2,vy3,vy4#Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4#,ro1,ro2
        print a.fitted_params2
        #initial_guess = [a.r[8], a.r[9], a.r[10], a.r[11]]
        #result3 = optimize.minimize(partial(IniLikelihood3,a), initial_guess,method = 'SLSQP',bounds=[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)])#method='nelder-mead')# method='Powell')#method = 'SLSQP')#method='nelder-mead')#method='COBYLA')#
        bounds=[(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax']))]#[(a.r[8],0.89),(a.r[9],0.89),(a.r[10],0.89),(a.r[11],0.89)]###[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)]    
        result3 = differential_evolution(partial(IniLikelihood3,a),bounds)     
        print result3
        if result3.success:
            fitted_params3 = result3.x
            print(fitted_params3)
        else:
            raise ValueError(result3.message)  
        #a.fitted_params3 = fitted_params3.tolist()

        Dx1 = fitted_params3[0]
        Dx2 = fitted_params3[1]
        Dx3 = fitted_params3[2]
        Dx4 = fitted_params3[3]
        a.fitted_params3 = Dx1,Dx2,Dx3,Dx4
        print a.fitted_params3

    #    initial_guess = [a.r[12], a.r[13], a.r[14], a.r[15]]
    #    result4 = optimize.minimize(partial(IniLikelihood4,a), initial_guess,method = 'SLSQP',bounds=[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)])#method='nelder-mead')# method='Powell')#method = 'SLSQP')#method='nelder-mead')#method='COBYLA')#
        bounds=[(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dymin']),float(ourinformation['dymax']))]#[(a.r[12],0.89),(a.r[13],0.89),(a.r[14],0.89),(a.r[15],0.89)]###[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)]    
        result4 = differential_evolution(partial(IniLikelihood4,a),bounds)
        print result4
        if result4.success:
            fitted_params4 = result4.x
            print(fitted_params4)
        else:
            raise ValueError(result4.message)  
        #a.fitted_params3 = fitted_params3.tolist()

        Dy1 = fitted_params4[0]
        Dy2 = fitted_params4[1]
        Dy3 = fitted_params4[2]
        Dy4 = fitted_params4[3]
        a.fitted_params4 = Dy1,Dy2,Dy3,Dy4

        print a.fitted_params4

    #    initial_guess = [a.r[16], a.r[17], a.r[18], a.r[19]]
        bounds=[(-0.999,0.999),(-0.999,0.999),(-0.999,0.999),(-0.999,0.999)]    
        result5 = differential_evolution(partial(IniLikelihood5,a),bounds) 
        print result5  
        #result5 = optimize.minimize(partial(IniLikelihood5,a), initial_guess,method = 'SLSQP',bounds=[(-0.999,0.999),(-0.999,0.999),(-0.999,0.999),(-0.999,0.999)])#method='nelder-mead')# method='Powell')#method = 'SLSQP')#method='nelder-mead')#method='COBYLA')#
        if result5.success:
            fitted_params5 = result5.x
            print(fitted_params5)
        else:
            raise ValueError(result5.message)  

        ro1 = fitted_params5[0]
        ro2 = fitted_params5[1]
        ro3 = fitted_params5[2]
        ro4 = fitted_params5[3]
        a.fitted_params5 = ro1,ro2,ro3,ro4
        print a.fitted_params5

        #bounds=[(0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0)]    
        #result6 = differential_evolution(partial(IniLikelihood6,a),bounds)    
        #print result6
        #if result6.success:
        #    fitted_params6 = result6.x
        #    print(fitted_params6)
        #else:
           # raise ValueError(result6.message)  

        gamma1 = a.newr[20]
        gamma2 = a.newr[21]
        gamma3 = a.newr[22]
        gamma4 = a.newr[23]
        #a.fitted_params6 = gamma1,gamma2,gamma3,gamma4
        #print a.fitted_params6

        a.par = vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4
        print a.par

        a.x=x
        a.y=y
        t=[a.t]
        ss2=[a.ss2]
        rescf = []
        for u in range(len(t)):
            a.t = t[u]
            a.ss2 = ss2[u]
            print a.t,a.ss2
            resacf = []
            loc = zip(x,y)
            #print loc
            resacf=np.array(multicore5(a,loc))
            resacf=np.transpose(resacf)
            #print resacf
            sumcf = 0
            #for i in resacf:
            #    sumcf = sumcf +np.array(i)
            rescf.append(resacf)
            print rescf
        scf = np.array(rescf)
        scf = scf/np.max(scf)

    elif ourinformation['Method'] == 'Best':
        pass 

    Resu = []
    x=a.x
    y=a.y 
    t=[a.t]
    ss2 = [a.ss2]
    print t, ss2
    for u in range(len(t)):
        a.t = t[u]
        a.ss2 = ss2[u]
        print a.ss2, a.t
        resaa = []
        loc = zip(x,y)
        print loc
        resaa=np.array(multicore4(a,loc))
        resaa=np.transpose(resaa)
        sum = 0
        #for i in resaa:
        #    sum = sum + numpy.array(i)
        Resu.append(resaa)
    Result = np.array(Resu)
    Result = Result/np.max(Result)

    progressBar.setValue(75)
    
    # for i in range(len(t)):
    #     plt.figure(i)
    #     plt.contourf(Ya,Xa,Result[0].reshape(len(Xa),len(Ya)))
    #     plt.grid()
    #     #plt.plot(-93.4683,29.205,'ro',ms=3)
    #     plt.colorbar()

    #     plt.show()
    coni = [n*10 for n in a.DLcon]

    xkm = Xa 
    ykm = Ya
    xaxis = []
    newx = np.ones([len(xkm)])
    for i in range(len(xkm)):
        newx[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[i],ykm[0])).km
        xaxis.append(newx[i])
    print xaxis
    yaxis = []
    newy = np.ones([len(xkm)])
    for i in range(len(ykm)):
        newy[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],ykm[i])).km
        yaxis.append(newy[i])
    print yaxis
    SXkm = []
    newxfield = np.ones([len(a.lati)])
    for i in range(len(a.longi)):
        newxfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],a.longi[i])).km 
        SXkm.append(newxfield[i])
    print SXkm
    SYkm = []
    newyfield = np.ones([len(a.lati)])
    for i in range(len(a.longi)):
        newyfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(a.lati[i],ykm[0])).km 
        SYkm.append(newyfield[i])
    print SYkm

    lon0km = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],lon0)).km 
    lat0km = geopy.distance.distance((xkm[0],ykm[0]),(lat0,ykm[0])).km 

    xcont = Xa
    ycont = Ya

    db = oceansdb.ETOPO()
    dcont = db['topography'].extract(lat=xcont, lon=ycont)
    decont = dcont['height']
    
    for i in range(len(t)):
        # plt.clf()
        plt.figure()
        plt.rcParams['font.size'] = 13   # change the font size of colorbar
        plt.rcParams['font.weight'] = 'bold' # make the test bolder        
        # print "xa",Xa
        # print "Ya",Ya
        if ourinformation['Method'] == 'Best':
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)))
                plt.colorbar()                 
                if ourinformation['Plot'] =="nofield":
                    pass
                elif ourinformation['Plot']=="field":
                    plt.scatter(a.longi,a.lati,s=coni)                                      
                if ourinformation['contour'] =="nocontour":
                    pass                
                elif ourinformation['contour'] == 'contour':
                    cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10)                        

            elif ourinformation['Map'] == 'km': 
                plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)))
                plt.colorbar()
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=coni)
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":
                    cs = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10)

        elif ourinformation['Method'] == 'Minimum':                
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)))
                plt.colorbar() 
                plt.contour(Ya,Xa,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])      
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.longi,a.lati,s=coni)
                elif ourinformation['Plot'] =="nofield":  
                    pass
                if ourinformation['contour'] =='contour':
                    cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10)    
                elif ourinformation['contour'] =="nocontour":
                    pass
            elif ourinformation['Map'] == 'km':
                plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)))
                plt.colorbar()
                plt.contour(xaxis,yaxis,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=coni)
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":
                    cs = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10)


        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        fffilename  = "Results/sunken"+time+"_back.png"
        plt.savefig(fffilename)
        plt.clf()
        img = cv2.imread(fffilename)
        crop_img = img[40:440,500:580]
        # cv2.imshow("image",crop_img)
        fffilename1 = "Results/sunken"+time+"_legend.png"
        cv2.imwrite(fffilename1,crop_img)
        plt.clf()

        if ourinformation['Method'] == 'Best':                     
            if ourinformation['Map'] == 'Coordinate':    
                plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)))
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
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
                    plt.scatter(a.longi,a.lati,s=coni)
                elif ourinformation['Plot'] == 'nofield':
                    pass
                if ourinformation['contour'] =='contour':
                    cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10) 
                elif ourinformation['contour'] =='nocontour':
                    pass

            if ourinformation['Map'] == 'km': 
                plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)))
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
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
                plt.text(0.2,1.8,"%d,%d"%(ourinformation['lat'],ourinformation['lon']),fontsize = 10, weight = 'normal', color ='r')
                plt.text(0.2,1.0,"0 km,0 km",fontsize = 10, weight = 'normal', color ='r')         
                plt.plot(0,0,'ro',ms=20)                
                if ourinformation['Plot'] == 'field':
                    plt.scatter(SXkm,SYkm,s=coni)#,marker=r'$\clubsuit$')           
                elif ourinformation['Plot'] == 'nofield':
                    pass
                if ourinformation['contour'] =='contour':
                    cs = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10) 
                elif ourinformation['contour'] =='nocontour':
                    pass

        elif ourinformation['Method'] == 'Minimum':
            if ourinformation['Map'] == 'Coordinate':   
                plt.contourf(Ya,Xa,Result[i].reshape(len(Xa),len(Ya)))
                plt.contour(Ya,Xa,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
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
                    plt.scatter(a.longi,a.lati,s=coni)
                elif ourinformation['Plot'] == 'nofield':
                    print "nofield"
                if ourinformation['contour'] =='contour':
                    cs = plt.contour(ycont,xcont,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10) 
                elif ourinformation['contour'] =='nocontour':
                    print 'nocontour'

            if ourinformation['Map'] == 'km': 
                plt.contourf(xaxis,yaxis,Result[i].reshape(len(Xa),len(Ya)))
                plt.contour(xaxis,yaxis,scf[i].reshape(len(Xa),len(Ya)), levels=[float(ourinformation["level"])], colors=['g'])
                plt.box(on=0)
                plt.grid(color = 'w', linestyle='-', linewidth=1)
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
                plt.text(0.2,1.8,"%d,%d"%(ourinformation['lat'],ourinformation['lon']),fontsize = 10, weight = 'normal', color ='r')
                plt.text(0.2,1.0,"0 km,0 km",fontsize = 10, weight = 'normal', color ='r')         
                plt.plot(0,0,'ro',ms=20)                
                if ourinformation['Plot'] == 'field':
                    plt.scatter(SXkm,SYkm,s=coni)#,marker=r'$\clubsuit$')
                    print "I am field"             
                elif ourinformation['Plot'] == 'nofield':
                    print "nofield"
                if ourinformation['contour'] =='contour':
                    cs = plt.contour(xaxis,yaxis,decont,cmap=plt.get_cmap('hot'))
                    plt.clabel(cs, inline=0.5,fontsize=10) 
                elif ourinformation['contour'] =='nocontour':
                    print 'nocontour'           

        #pyplot.text(self.lon0-0.005, self.lat0-0.005, "x", fontsize = 16, weight = 'bold', color ='m')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        filename  = "Results/submerged"+time+".png"
        # plt.show()
        plt.savefig(filename, dpi=599, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False)
        figPGW = np.array([(float(ourinformation['x_max']) - float(ourinformation['x_min']))/4792.0, 0.0, 0.0, -(float(ourinformation['y_max']) - float(ourinformation['y_min']))/3600.0, float(ourinformation['x_min']), float(ourinformation['y_max'])])
        #figPGW = np.array([(float(Xticks[len(Xticks)-1]) - float(Xticks[0]))/4792.0, 0.0, 0.0, -(float(Yticks[len(Yticks)-1]) - float(Yticks[0]))/3600.0, float(Xticks[0]), float(Yticks[len(Yticks)-1])])
        filename11 = "Results/submerged"+time+".pgw"
        figPGW.tofile(filename11 , sep = '\n', format = "%.16f")
        return filename,fffilename1

    # plt.show()

