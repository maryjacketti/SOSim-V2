# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from math import *
import numpy as np
import random
import utm
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import integrate
from  scipy.optimize import differential_evolution
from scipy.stats import chi2
import pandas as pd
import datetime
from multiprocessing import pool
import multiprocessing as mp
from multiprocessing import cpu_count
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
import oceansdb
import seawater as sw
from seawater.library import T90conv
import itertools
import seaborn as sns
from mpl_toolkits import mplot3d
from netCDF4 import Dataset
import netCDF4
import numpy.ma as ma 
import cv2
import geopy.distance

ourinformation = {}

def ff(x,y,mux,muy,Dx,Dy,ro):
    s=1./(2*np.pi*Dx*Dy*np.sqrt(1-ro**2))*np.exp(-1/(2*(1-ro**2))*((x-mux)**2/Dx**2+(y-muy)**2/Dy**2-2*ro*(x-mux)*(y-muy)/(Dx*Dy)))#stats.norm(loc=mu,scale=Dx).pdf(x)
    return s    

def fcont(x,y,vx,vy,Dx,Dy,ro,x0,y0,t,s):
    k=0
    for i in range(len(s)):
        mux = x0 + vx*(t-s[i])
        muy = y0 + vy*(t-s[i])
        sx = np.sqrt(2.0*Dx*(t-s[i]))+0.0001
        sy = np.sqrt(2.0*Dy*(t-s[i]))+0.0001
        mm = ff(x,y,mux,muy,sx,sy,ro)
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

def IniLikelihood(a,parameter):
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    #print vx1
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    #print st
    if a.ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        st = np.concatenate((a.st,a.priort),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
    
    elif a.ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        st = a.st
        s=a.TTi     

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])

    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    #c = np.exp((100-Lamda)/b)/(1*np.exp((100-Lamda)/b))-np.exp((-Lamda)/b)/(1+np.exp((-Lamda)/b))
                    #if c > 1e-308:
                    IniIndLikelihood[ci] = np.log(Lamda)-Lamda*DLcon[ci]#*(1.0/c)
                else:
                    IniIndLikelihood[ci] = 0
                    Lamda = 0

    for i in range(1):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci] == 0:
                    CompLikelihood = 0

    return IniIndLikelihood
            #print MaxLogLike

def integ(a,loc):
    parameter = a.r
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    [x,y]=loc  
    t=a.t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    s=a.ti
    ProObsGivenPar = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,t,s) \
    +gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,t,s) \
    +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,t,s) \
    +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,t,s)
    return ProObsGivenPar

def integcf(a,loc):
    parameter = a.par
    vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    ConResult = 0
    [x,y]=loc
    t=a.t
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    s = a.ti
    ProObsGivenPar = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,x0,y0,t,s) \
    +gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,x0,y0,t,s) \
    +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,x0,y0,t,s) \
    +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,x0,y0,t,s)
    return ProObsGivenPar

def depth(a,locs):
    delta = 25  # default 
    depthas = np.arange(900,1200,delta)  # change the depth later
    [xd,yd] = locs
    print xd
    db = oceansdb.CARS()
    zmax=[]
    zmin =[]
    salinityas=[]
    temperas=[]
    densas=[]
    densityas = []
    for j in range(len(depthas)):
        salias = db['sea_water_salinity'].extract(var='mean',doy=200,depth=depthas[j],lat=xd,lon=yd)
        aa = float(salias['mean'])            
        salinityas.append(aa)                     
        tempas = db['sea_water_temperature'].extract(var='mean',doy=200,depth=depthas[j], lat=xd, lon=yd)
        b = float(tempas['mean'])
        temperas.append(b)
    temperatureas = T90conv(temperas)
    densityas = sw.dens0(salinityas,temperatureas)
    for k in range(len(densityas)):
        if densityas[k] >=1027.52 and densityas[k] <= 1027.77:
            densas.append(densityas[k])
    maxden = np.max(densas)
    e = list(densityas).index(maxden)        
    minden = np.min(densas)
    f = list(densityas).index(minden)
    depthmax = depthas[e]
    print "depthmax",depthmax
    depthmin = depthas[f]
    print "depthmin",depthmin
    return depthmax,depthmin

def all_same(items):
    return all(x==items[0] for x in items)

def Likelihood(a,N):
    print "This is Likelihood !"
    Result = []
    parameter=sampler(N)
    IniLikelihood=np.array(multicore1(a,parameter))
    IniIndLikelihood = np.transpose(IniLikelihood)
    if a.ourinformation['SubmergedType'] == 'OSCAR':
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        print "DLx=",DLx
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        print "DLy=",DLy
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        print "DLcon=",DLcon
    
    elif a.ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 

    CompLikelihood =np.ones(N)
    FileLikelihood = np.zeros(N)
    Likelihoodi = np.zeros(N)  
   # print t 
    for i in range(N):
        for ci in range(len(DLx)):
            if DLcon[ci]>0:
                if IniIndLikelihood[ci,i] == 0:
                    CompLikelihood[i] = 0

    MaxLogLike=-22
    for i in range(N):
        for ci in range(len(a.DLx)):
            if DLcon[ci]>0:
                if CompLikelihood[i]==1:
                    FileLikelihood[i] =  FileLikelihood[i] + IniIndLikelihood[ci,i]
        if CompLikelihood[i]==1:
            if MaxLogLike ==-22:
                MaxLogLike=FileLikelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,FileLikelihood[i]])
    print MaxLogLike

    for i in range(N):
        if CompLikelihood[i]==1:
            FileLikelihood[i] = FileLikelihood[i] - MaxLogLike +7
            Likelihoodi[i] = np.exp(FileLikelihood[i])
    return Likelihoodi,MaxLogLike

#------------------confidence bounds--------------
def IniLikelihood1(a,parameter):
    vx1,vx2,vx3,vx4= parameter
    Max = a.MaxLogLike
    r = a.r
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0    
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
    Dx2= r[9]
    Dx3 = r[10]
    Dx4 = r[11]        
    Dy1 = r[12]
    Dy2 = r[13]
    Dy3 = r[14]
    Dy4 = r[15]
    if a.ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif a.ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
        if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #print login 
    return login 

def IniLikelihood2(a,parameter):
    #gamma1,gamma2,gamma3,gamma4,ro1,ro2,ro3,ro4,Dx4,Dy1,Dy2= parameter
    params = a.fitted_params
    vy1,vy2,vy3,vy4 = parameter
    vx1,vx2,vx3,vx4 = params
    Max = a.MaxLogLike
    r = a.r
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19]   
    Dx1 = r[8]
    Dx2= r[9]
    Dx3 = r[10]
    Dx4 = r[11]        
    Dy1 = r[12]
    Dy2 = r[13]
    Dy3 = r[14]
    Dy4 = r[15]
    if a.ourinformation['SubmergedType'] == 'OSCAR'or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif a.ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
       if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #print login 
    return login 

def IniLikelihood3(a,parameter):
    Dx1,Dx2,Dx3,Dx4 = parameter
    params = a.fitted_params
    vx1,vx2,vx3,vx4 = params
    params2 = a.fitted_params2
    vy1,vy2,vy3,vy4 = params2
    Max = a.MaxLogLike
    r = a.r    
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
    if ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0 

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
       if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
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
    Max = a.MaxLogLike
    r = a.r
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    ro1 = r[16]
    ro2 = r[17]
    ro3 = r[18]
    ro4 = r[19] 
    if ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
       if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
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
    Max = a.MaxLogLike
    r = a.r
    gamma1 = r[20]
    gamma2 = r[21]
    gamma3 = r[22]
    gamma4 = r[23]
    if ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
       if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
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
    Max = a.MaxLogLike
    if ourinformation['SubmergedType'] == 'OSCAR' or a.ourinformation['SubmergedType']=='GNOME' or a.ourinformation['SubmergedType']=='Other':    
        DLx = np.concatenate((a.DLx,a.priorx),axis = None)
        DLy = np.concatenate((a.DLy,a.priory),axis = None)
        DLcon = np.concatenate((a.DLcon,a.priorcon),axis = None)
        s = np.concatenate((a.TTi,a.ppti),axis = 0)
        st = np.concatenate((a.st,a.priort),axis = None)        
    elif ourinformation['SubmergedType'] == 'Nodate':
        DLx = a.DLx
        DLy = a.DLy
        DLcon = a.DLcon 
        s=a.TTi
        st = a.st
    x0=a.x0
    y0=a.y0
    Dx0=a.Dx0
    Dy0=a.Dy0
    r = a.r
    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])
    # print len(x),range(31)
    for ci in range(len(DLx)):
       if DLcon[ci] >0:
            for i in range(1):
                ss=s[ci]
                ss=ss[ss<st[ci]]
                Prob = gamma1*fcont(DLx[ci],DLy[ci],vx1,vy1,Dx1,Dy1,ro1,x0,y0,st[ci],ss) \
                 +gamma2*fcont(DLx[ci],DLy[ci],vx2,vy2,Dx2,Dy2,ro2,x0,y0,st[ci],ss) \
                 +gamma3*fcont(DLx[ci],DLy[ci],vx3,vy3,Dx3,Dy3,ro3,x0,y0,st[ci],ss) \
                 +gamma4*fcont(DLx[ci],DLy[ci],vx4,vy4,Dx4,Dy4,ro4,x0,y0,st[ci],ss)
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
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    return login 
#-----------------confidence bounds--------------------------
def multicore1(a,parameter):
        pool = mp.Pool(19)
        res = pool.map(partial(IniLikelihood,a),parameter)
        return res

def multicore2(a,loc):
        pool = mp.Pool(19)
        res = pool.map(partial(integ,a),loc)
        return res

def multicore3(a,loc):
        pool = mp.Pool(19)
        res = pool.map(partial(integcf,a),loc)
        return res

def multicore4(a,locs):
        pool = mp.Pool(19)
        res = pool.map(partial(depth,a),locs)
        return res

def extract_key(v):
        return v[0]

class Preliminars: # SOSim
    def __init__(self): 
        self.w = 4
        self.u = self.w + 1
        self.delta = 0
        self.args = []
        self.discarded = 0.0
        self.valid = 0.0
        self.GammaPossible = []

class soscore(Preliminars):
    def __init__(self,datalist):
        Preliminars.__init__(self)
        lat = []
        lat.append(datalist['lat'])
        # lon = datalist['lon'] # longitude 
        lon = []
        lon.append(datalist['lon'])
        SpillT = []        
        SpillT.append(datalist['starttime'])
        SpillT.append(datalist['endtime'])
        print "start",SpillT
        # SpillT = datalist['SpillTime']  # user's spill start and spill end #SpillT[0] is the spill end time; SpillT[1] is spill start time
        # SampleT = datalist['SampleTime']
        PredictT = []
        PredictT.append(datalist['PredictTime'])
        OilType = []
        OilType.append(datalist['OilType'])
        # Scale = datalist['Scale']
        # Node = datalist['Node']
        Scale = []
        Scale.append(float(datalist['lonscale']))
        Scale.append(float(datalist['latscale']))

        print datalist
        print "_____________________This is soscore_____________________________"

        Node = []
        Node.append(float(datalist['xNode']))
        Node.append(float(datalist['yNode']))

        print Node

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
        Count = 10
        #define spill point
        coord0 = utm.from_latlon(lat, lon)
        x0 = coord0[0]/1000.0
        y0 = coord0[1]/1000.0

        durationt = [CalTime(SpillT[0],SpillT[1])]
        dura = np.pad(durationt,(0,1),'constant')
        #print dura
#        afterspill = [CalTime(SampleT[0],SpillT[1])]
#        beforeend = [CalTime(SpillT[0],SampleT[0])]
        PTi = []
        t = [CalTime(SpillT[0],PredictT[vld]) for vld in range(len(PredictT))]
        #print t 
        for i in range(len(t)):
            after = [CalTime(PredictT[i],SpillT[1])]            
            if dura[0]!=dura[1] and after > 0:
                Ti = np.linspace(0,t[i],Count)
                PTi.append(Ti)                
            elif SpillT[0] < SpillT[1]:
                Ti = np.linspace(dura[1],dura[0],Count)
                PTi.append(Ti)   
            else:
                # Ti = np.zeros(self.Count)
                Ti = np.zeros(Count)
                PTi.append(Ti)   
#        x0=np.pad(x0, (0,3), 'constant')
#        y0=np.pad(y0, (0,3), 'constant')       
        PTi = np.array(PTi) 
        #print t 
        #print PTi 
        self.x0 = x0
        self.y0 = y0
        self.SpillT = SpillT
        self.PredictT = PredictT
        self.Scale = Scale
        # self.Node = Node
        self.xNode = Node[0]
        self.yNode = Node[1]
        self.OilType = OilType
        self.Dx0 = sigmax0
        self.Dy0 = sigmay0
        self.x00 = lat
        self.y00 = lon
        self.t = t # t is prediction time 
        self.scale = Scale
        self.lat0 = lat[0]
        self.lon0 = lon[0]
        self.Ti = PTi  # Ti is the duration of prediction time divide into different parts 
        self.dura = dura
        self.Count = Count
        #Load campaign data

    def UploadCampaign(self,CampaignFileName):  # upload Campaign part
        Preliminars.__init__(self)
        DLx = []
        DLy = []
        DLcon = []
        st = [] # sample time
        salinity = []
        temper = []
        bat = []
        dens = []
        DLlatselected = []
        DLlonselected = []
        DLcselected = []
        depthselected = []
        uniqDLlat = []
        DLcl = []
        f=[]
        lati=[]
        longi=[]
        dep = []
        s = []
        # de = []
        sst = []
        Modellat = []
        Modellon = []
        db = oceansdb.CARS()
        de = oceansdb.ETOPO()
        for i in range(len(CampaignFileName)):
            if i == 0:
                campdata = pd.read_csv(CampaignFileName[i])
            else:
                campdata = pd.concat([campdata , pd.read_csv(CampaignFileName[i])],axis = 0,ignore_index=True)                

        SampleT = np.array(campdata['SampleTime'])    
        DLlat = np.array(campdata["Lat"])
        DLlon = np.array(campdata["Lon"])
        DLc = np.array(campdata["Con"])
        depth = np.array(campdata["depth"])
        s.extend(SampleT)
#------------------filter the same position data------    
        for i in range(len(DLlat)):
            c = round(DLlat[i],4)
            d = round(DLlon[i],4)
            Modellat.append(c)
            Modellon.append(d)  

        coo = zip(Modellat,Modellon)
        coocon = zip(coo,DLc,SampleT,depth)
        extract_key = lambda x:x[0]
        datap = sorted(coocon,key=extract_key)       
        resultp = [[k,[x[1:4] for x in g]] for k, g in itertools.groupby(datap, extract_key)]

        DLclp=[]
        fp=[]
        ttp=[]
        conc = []
        for i in range(len(resultp)):
            conres=[list(p) for m, p in itertools.groupby(resultp[i][1],lambda x:x[1])]
            for j in conres:
                #print j
                jarray=np.array(j)
                con = np.mean(map(float,jarray[:,0]))
                DLclp.append(con)            
                h = resultp[i][0]
                fp.append(h)
                ttp.append(j[0][1])
                dep.append(j[0][2])
        fp = np.array(fp)     # location 
        DLclp = np.array(DLclp)   # average concentration
        ttp = np.array(ttp)  #samplet
        dep = np.array(dep)

        Dllat = fp[:,0]
        Dllon = fp[:,1]
        Dllat = np.array(Dllat)
        Dllon = np.array(Dllon)
#------------------filter the same position data------  
        for i in range(len(dep)):
            sali = db['sea_water_salinity'].extract(var='mean',doy=200,depth=depth[i],lat=DLlat[i],lon=DLlon[i])          
            temp = db['sea_water_temperature'].extract(var='mean',doy=200,depth=depth[i], lat=DLlat[i], lon=DLlon[i])
            d = de['topography'].extract(lat=DLlat[i], lon=DLlon[i])
            # sali = db['sea_water_salinity'].extract(var='mean',doy=200,depth=dep[i],lat=Dllat[i],lon=Dllon[i])          
            # temp = db['sea_water_temperature'].extract(var='mean',doy=200,depth=dep[i], lat=Dllat[i], lon=Dllon[i])
            # d = de['topography'].extract(lat=Dllat[i], lon=Dllon[i])
            bathy = d['height']
            bath = abs(bathy)
            bath = np.array(bath)
            a1 = float(sali['mean'])
            b = float(temp['mean'])
            salinity.append(a1)
            temper.append(b)
            bat.append(bath)   

        for j in range(len(s)):  
            cc = CalTime(self.SpillT[0],s[j])
            st.append(cc)  
        position = np.array(zip(DLlon,DLlat))
        temperature = T90conv(temper)
        density = sw.dens0(salinity,temperature)
        d = zip(DLlat,DLlon,DLc,depth,density,bat)
        print "bathy=", bat 
        for i in range(len(density)):
            if 1027.52 <= density[i] <= 1027.77:
                dens.append(density[i])
        DLlatselected = []
        DLlonselected = []
        DLcselected = []
        depthselected = []
        bathyselected = []
        for i in range(len(dep)):
            for j in range(len(dens)):
                if d[i][4] == dens[j]:
                    DLlatselected.append(d[i][0])
                    DLlonselected.append(d[i][1])
                    DLcselected.append(d[i][2])
                    depthselected.append(d[i][3])
                    bathyselected.append(d[i][5])
        bathyselected = np.array(bathyselected)
        print "bathyselected",bathyselected      
        bathymin = np.min(bathyselected)
        bathymax = np.max(bathyselected)
        coordinate = zip(DLlatselected,DLlonselected)
        coordcon = zip(coordinate,DLcselected)
        data = sorted(coordcon,key=extract_key)
        result = [[k,[x[1] for x in g]] for k, g in itertools.groupby(data, extract_key)]
        for i in range(len(result)):
            con = np.mean(result[i][1])
            DLcl.append(con)
            h = result[i][0]
            f.append(h)
        for i in range(len(f)):
            l=f[i][0]
            t=f[i][1]
            lati.append(l)
            longi.append(t)
        
        # uniqcoordinate = list(k for k, _ in itertools.groupby(sorted(coordinate)))
        #print density
        N = len(DLlat)
        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(lati,longi)])
        DLx.append(np.array(map(float,camdatalist[:,0]))/1000)
        #print DLx
        DLy.append(np.array(map(float,camdatalist[:,1]))/1000)
#            st.append(CalTime(self.SpillT[0],SampleT[0]))             
        for s in DLcl:
            if s == 0.0:
                conValue = (0.1/1000)
            else:
                conValue = (s/1000)
            DLcon.append(conValue)
        STti=[]

        for k in range(len(st)):
            if self.dura[0]!=self.dura[1] and st[k] < self.dura[0]:
                TTi = np.linspace(0,st[k],self.Count)
                STti.append(TTi)                
            elif self.SpillT[0] < self.SpillT[1]:
                TTi = np.linspace(self.dura[1],self.dura[0],self.Count)
                STti.append(TTi)                 
            else:
                TTi = np.zeros(self.Count)
                STti.append(TTi)

        STti = np.array(STti)
        tt = np.array(st)
        DLx[0] = np.array(DLx[0])
        DLy[0] = np.array(DLy[0])  
        self.DLx = DLx[0]
        self.DLy = DLy[0]
        self.DLcon = np.array(DLcon)
        self.st = tt
        self.TTi = STti  #TTi is a sample time divided into different parts
        self.N = N
        self.DLlat = np.array(DLlat)
        self.DLlon = np.array(DLlon)
        self.position = position
        self.bathy = bathyselected
        self.bathymin = bathymin
        self.bathymax = bathymax

    def upprior(self,Files):    # excel data upload
        Mlat = []
        Mlon = []
        Mcon = []
        Ms = []
        Mde =[]
        afterspill = []
        mt = []
        priorcon =[]
        index = []
        realdata = []
        Modellat =[]
        Modellon =[]
        xr = []
        yr = []
        xxx =[]
        yyy = [] 
        da=[]
        mmt = []
        conc=[]        
        FileName = []
        FileName.append(Files)
        #print len(FileName)       
        for ii in range(len(FileName)):
            priordata = pd.read_csv(FileName[ii])
            #print priordata
            lat = np.array(priordata['lat'])
            lon = np.array(priordata['lon']) 
            depth = np.array(priordata["depth"])            
            ModelT =  np.array(priordata['predicttime'])   
            Modelc =  np.array(priordata['concentration'])
            Mlat.extend(lat)
            Mlon.extend(lon)
            Mcon.extend(Modelc)
            Ms.extend(ModelT)
            Mde.extend(depth) 

        for j in range(len(ModelT)):
            after = CalTime(ModelT[j],self.SpillT[1]) 
            beforeend = CalTime(self.SpillT[0],ModelT[j]) 
            mt.append(beforeend)       
            afterspill.append(after)
        mt = np.array(mt)
        #print mt 
        for i in range(len(Mlat)):
            c = round(Mlat[i],4)
            d = round(Mlon[i],4)
            Modellat.append(c)
            Modellon.append(d)

        data = zip(Modellat,Modellon,Modelc)
        coordr = zip(Modellat,Modellon)   
        n = ceil(np.sqrt(self.N))-1
        size = int(self.N/(n*n)) +1
        #print size 
        X = np.linspace((self.lat0)-self.scale[0]-0.005,(self.lat0)+self.scale[0]+0.005,n+1)
        Y = np.linspace((self.lon0)-self.scale[1]-0.005,(self.lon0+0.005)+self.scale[1],n+1) 
        rdata = []
        arr=[[[] for arx in range(len(X)-1)] for ary in range(len(Y)-1)]
        clat = []
        clong = []
        ccon = []
        ct = []
        for p in range(len(Mlon)):
            if Mlon[p]>=Y[0] and Mlon[p]<=Y[-1] and Mlat[p]>=X[0] and Mlat[p]<=X[-1]:
                l=int((Mlat[p]-X[0])/(2*(self.scale[0]+0.005)/n))
                m=int((Mlon[p]-Y[0])/(2*(self.scale[1]+0.005)/n))
                #print l,m
                clat.append(Mlat[p])
                clong.append(Mlon[p])
                ccon.append(Mcon[p])
                ct.append(mt[p])

        coo = zip(clat,clong)
        coocon = zip(coo,ccon,ct)
        extract_key = lambda x:x[0]
        datap = sorted(coocon,key=extract_key)       
        resultp = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datap, extract_key)]

        DLclp=[]
        fp=[]
        ttp=[]
        conc = []
        for i in range(len(resultp)):
            conres=[list(p) for m, p in itertools.groupby(resultp[i][1],lambda x:x[1])]
            for j in conres:
                #print j
                jarray=np.array(j)
                con = np.mean(jarray[:,0])
                DLclp.append(con)            
                h = resultp[i][0]
                fp.append(h)
                ttp.append(j[0][1])
        fp = np.array(fp)
        DLclp = np.array(DLclp)
        ttp = np.array(ttp)

        for p in range(len(DLclp)):
            if Mlon[p]>=Y[0] and Mlon[p]<=Y[-1] and Mlat[p]>=X[0] and Mlat[p]<=X[-1]:
                l=int((Mlat[p]-X[0])/(2*(self.scale[0]+0.005)/n))
                m=int((Mlon[p]-Y[0])/(2*(self.scale[1]+0.005)/n))
                arr[l][m].append((fp[p][0],fp[p][1],DLclp[p],ttp[p]))  # change later 

# sampling plan 
        for l in range(len(X)-1):
            for m in range(len(Y)-1):
                #print arr[l][m]
                if len(arr[l][m]) <= size:
                    rdata.append(arr[l][m])
                else:
                    ind=np.random.choice(range(len(arr[l][m])),size,replace=True)
                    tmp = np.array(arr[l][m])
                    rdata.append(tmp[ind])
        for g in range(len(rdata)):
            for o in range(len(rdata[g])):
                da.append(rdata[g][o])
        rdata = np.array(da)

        for b in range(len(rdata)):
            xx = rdata[b][0]
            yy = rdata[b][1]
            lt = rdata[b][3]
            xxx.append(xx)
            yyy.append(yy)
            mmt.append(lt)
        mmt = np.array(mmt)
        print mmt 
        print xxx
        print yyy
        #coord = np.array([utm.from_latlon(kk,mm) for (kk,mm) in zip(xxx,yyy)])
        coord = np.array([utm.from_latlon(kk,mm) for (kk,mm) in zip(Mlat,Mlon)])
        print coord
        priorx.append(np.array(map(float,coord[:,0]))/1000)
        priory.append(np.array(map(float,coord[:,1]))/1000)

        for i in range(len(rdata)):
            if rdata[i][2] == 0.0:
                conValue = (0.1/1000)
            else:
                conValue = (rdata[i][2]/1000)
            priorcon.append(conValue)
        priorcon = np.array(priorcon)
        pti = np.linspace(self.dura[1],tt[2],self.Count)
        ppti = np.array([pti]*len(xxx))

        self.priorx = priorx[0]
        self.priory = priory[0]
        self.priorcon = priorcon
        self.ppti = ppti   # Tti is prior time divided into different parts 
        self.priort = mmt

    def upNetCDF(self,File):  # when click Submerged hydrodynamic Upload, OSCAR part 
        # read NetCDF 
        data = Dataset(File,mode='r')
        px = data.variables['longitude'][:]
        py = data.variables['latitude'][:]
        [px,py] = np.meshgrid(px,py)
        Z = data.variables['total_concentration'][:]
        pt = data.variables['time'][:]
        dep=data.variables['depth'][:]

        index=[]
        for i in range(Z.shape[0]):
            ind1=[]
            for j in range(Z.shape[1]):
                PP = Z[i,j,:,:]
                PP=ma.masked_values(PP,-99)
                ind = np.where(PP>0)
                ind1.append(ind)
            index.append(ind1)
        #index = np.array(index)
        # valid value: concentration not equal to zero  
        priorlon = []
        priorlat = []
        priorcon = []
        priord = []
        priordep=[]
        priorT=[]
        Mllat = []
        Mllon = []
        da = []
        xxx =[]
        yyy = []
        mmt =[]
        tt = np.zeros(len(pt))
        for k in range(len(pt)):
            tt[k] = (pt[k]/86400.) + 44.        
        #for j in range(len(index)):
        #    for m in range(len(index[j])): #for m in [3]:
        for j in [2]:    # change later  j for time 
            for m in [15]:   # change later  m for depth 
                zl = px[index[j][m]]
                zla = py[index[j][m]]
                zcon = Z[j,m][index[j][m]]
                priordep.append(dep[m])
                priorT.append(pt[j])
                priorlon.append(zl)   # all depth available lon; len(Time)*len(depth)
                priorlat.append(zla)
                priorcon.append(zcon)  # first 20 (Depth len) is the first Time 

        for i in range(len(priorlon[0])):
            c = round(priorlon[0][i],4)
            d = round(priorlat[0][i],4)
            Mllat.append(c)
            Mllon.append(d) 

        data = zip(Mllat,Mllon,priorcon[0])
        coordr = zip(Mllat,Mllon)   
        n = ceil(np.sqrt(self.N))-1
        size = int(self.N/(n*n)) +1
        #print size 
        X = np.linspace((self.lat0)-self.scale[0]-0.005,(self.lat0)+self.scale[0]+0.005,n+1)
        Y = np.linspace((self.lon0)-self.scale[1]-0.005,(self.lon0+0.005)+self.scale[1],n+1) 
        rdata = []
        arr=[[[] for arx in range(len(X)-1)] for ary in range(len(Y)-1)]

        clat = []
        clong = []
        ccon = []
        ct = []
        conc = []
        for p in range(len(priorlon[0])):
            if priorlon[0][p]>=Y[0] and priorlon[0][p]<=Y[-1] and priorlat[0][p]>=X[0] and priorlat[0][p]<=X[-1]:
                l=int((priorlat[0][p]-X[0])/(2*(self.scale[0]+0.005)/n))
                m=int((priorlon[0][p]-Y[0])/(2*(self.scale[1]+0.005)/n))
                #print l,m
                clat.append(priorlat[0][p])
                clong.append(priorlon[0][p])
                ccon.append(priorcon[0][p])
                ct.append(tt[2])  # change later time 

        coo = zip(clat,clong)
        coocon = zip(coo,ccon,ct)
        extract_key = lambda x:x[0]
        datap = sorted(coocon,key=extract_key)       
        resultp = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datap, extract_key)]

        DLclp=[]
        fp=[]
        ttp=[]
        for i in range(len(resultp)):
            conres=[list(p) for m, p in itertools.groupby(resultp[i][1],lambda x:x[1])]
            for j in conres:
                #print j
                jarray=np.array(j)
                con = np.mean(jarray[:,0])
                DLclp.append(con)            
                h = resultp[i][0]
                fp.append(h)
                ttp.append(j[0][1])
        fp = np.array(fp)
        DLclp = np.array(DLclp)
        ttp = np.array(ttp)

        for p in range(len(DLclp)):
            if priorlon[0][p]>=Y[0] and priorlon[0][p]<=Y[-1] and priorlat[0][p]>=X[0] and priorlat[0][p]<=X[-1]:
                l=int((priorlat[0][p]-X[0])/(2*(self.scale[0]+0.005)/n))
                m=int((priorlon[0][p]-Y[0])/(2*(self.scale[1]+0.005)/n))
                arr[l][m].append((fp[p][0],fp[p][1],DLclp[p],ttp[p]))  # change later 

        #print clat 
        #print len(clat)
        #print arr 
        #print len(arr)
# sampling plan 
        for l in range(len(X)-1):
            for m in range(len(Y)-1):
                #print arr[l][m]
                if len(arr[l][m]) <= size:
                    rdata.append(arr[l][m])
                else:
                    ind=np.random.choice(range(len(arr[l][m])),size,replace=True)
                    tmp = np.array(arr[l][m])
                    rdata.append(tmp[ind])
        for g in range(len(rdata)):
            for o in range(len(rdata[g])):
                da.append(rdata[g][o])
        rdata = np.array(da)
        print "rdata=",rdata
        # X = np.linspace(28.64,28.74,n+1)
        # Y = np.linspace(-88.525,-88.325,n+1)  
        #print ym

        for b in range(len(rdata)):
            xx = rdata[b][0]
            yy = rdata[b][1]
            con = rdata[b][2]
            lt = rdata[b][3]
            xxx.append(xx)
            yyy.append(yy)
            conc.append(con)
            mmt.append(lt)
        mmt = np.array(mmt)
        conc = np.array(conc)
        priorx = []
        priory = []
        print "xxx=",xxx
        xxx = np.array(xxx)
        yyy =np.array(yyy)
        print "new xxx=",xxx        
            #print priorlat[2]
            #print priorlon[2]
            #print len(priorlat[2])
            #print len(priorlon[2])
        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(xxx,yyy)])
        #camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(priorlat[0],priorlon[0])])
        priorx.append(np.array(map(float,camdatalist[:,0]))/1000)
        priory.append(np.array(map(float,camdatalist[:,1]))/1000)
        pti = np.linspace(self.dura[1],tt[2],self.Count)
        ppti = np.array([pti]*len(xxx))

        print "priorcon=",conc
        self.priordistribution = zip(priorlon[0],priorlat[0])
        self.priorx = priorx[0]
        self.priory = priory[0]
        self.priorcon = conc#priorcon[0]
        self.priortime = tt[2]  # change later the time 
        self.priort = mmt
        self.pti = pti
        self.ppti = ppti
        self.priorlon = yyy
        self.priorlat = xxx
        self.priorT=priorT

    def GupNetCDF(self,File):  # when click Submerged hydrodynamic Upload, OSCAR part 
        # read NetCDF 
        data = Dataset(File,mode='r')
        px = data.variables['longitude']
        py = data.variables['latitude']
        print data.variables['longitude']
        particle = data.variables['particle_count'][:]
#        [px,py] = np.meshgrid(px,py)
        #Z = data.variables['total_concentration'][:]
        Z = data.variables['surface_concentration'][:]        
        pt = data.variables['time'][:]
        dep=data.variables['depth'][:]

        t_count=[]
        priorx = []
        priory = []        
        for i in range(len(particle)):
            t_count.extend([i]*particle[i])
        px=np.array(px)
        print px.shape
        py=np.array(py)
        Z = np.array(Z)
        t_count=np.array(t_count)

        tt = np.zeros(len(pt))
        for k in range(len(pt)):
            tt[k] = (pt[k]/86400.)

        px1=px[np.where(t_count==1)]
        py1=py[np.where(t_count==1)]
        con = Z[np.where(t_count==1)]
        print px1
        print py1
        print con

        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(py1,px1)])
        priorx.append(np.array(map(float,camdatalist[:,0]))/1000)
        priory.append(np.array(map(float,camdatalist[:,1]))/1000)
        pti = np.linspace(self.dura[1],tt[1],self.Count)

        self.priordistribution = zip(px1,py1)
        self.priorx = priorx[0]
        self.priory = priory[0]
        self.priorcon = con
        self.priortime = tt[1]
        self.pti = pti
        #print priordep[10]
        #self.priordep=priordep[10]
        #self.priorT=priorT

def submerged_main(myinformation,progressBar):
    #delta_gamma = 0.1

    # print "submerged!"
    # print myinformation

    global ourinformation

    ourinformation = myinformation
    # print ourinformation

    if 'HydroButton' in ourinformation and ourinformation['HydroButton'] != '':
        filename = ourinformation['HydroButton']
    else:
        filename = "null"
    
    print filename

    # a = soscore("datainput1.csv")
    a = soscore(ourinformation)
    a.ourinformation = ourinformation
    progressBar.setValue(15)
    a.UploadCampaign(myinformation['CampaignButton'])

    #a.retardationDueOilType()
    #a.x0y0DueSinkingRetardation() 
    print ourinformation
    a.myinformation = ourinformation
    # If The user clicked 'OSCAR'.
    # .nc file is uploaded
    if ourinformation['SubmergedType'] == 'OSCAR':
        a.upNetCDF(filename)
    # If The user clicked 'Other'.
    # .csv file is uploaded
    elif ourinformation['SubmergedType'] == 'Other':
        a.upprior(filename)
    # If The user clicked 'GNOME'.
    # .nc file is uploaded
    elif ourinformation['SubmergedType'] == 'GNOME':
        a.GupNetCDF(filename)

    # scatterplot field data
    #df = pd.DataFrame(a.position,columns=["x","y"])
    #sns.jointplot(x="x",y="y",data=df)
    # if ourinformation['SubmergedType'] == 'OSCAR':
    #     dff = pd.DataFrame(a.priordistribution,columns=["x","y"])
    #     sns.jointplot(x="x",y="y",data=dff)    
    #print a.DLx
    #print type(a.DLx)
    #print a.DLlat
    #print a.DLlon
    # plot concentration 
    #ax = plt.axes(projection='3d')
    #ax.scatter3D(a.DLlon, a.DLlat, a.DLcon)

    SX = a.DLx
    SY = a.DLy
    lat0 = a.lat0  # transform parameters 
    lon0 = a.lon0
    DLcon = a.DLcon

    #gridC = a.gridC
    # print DLcon

    # Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.Node+1)
    # Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.Node+1)
    print "scale",a.scale
    Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.xNode+1)
    Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.yNode+1)
    
    Xxa = np.linspace(a.lat0+a.scale[0],a.lat0-a.scale[0],a.xNode+1)
    Yya = np.linspace(a.lon0+a.scale[1],a.lon0-a.scale[1],a.yNode+1)
    [XXa,YYa] = np.meshgrid(Xxa,Yya)
    xxa = np.concatenate(XXa)
    yya = np.concatenate(YYa)

    coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(Xa,Ya)])
    xa = np.array(map(float,coord[:,0]))/1000
    ya = np.array(map(float,coord[:,1]))/1000
    [x,y] = np.meshgrid(xa,ya)
    x = np.concatenate(x)
    y = np.concatenate(y)

    X=SX
    Y=SY
    t = a.t
    SP = a.st
    Count = a.Count
    # t = a.ti
    # SP = a.Ti
#    print SP
#    print SP[0]

    Dx0 = 0.05
    Dy0 = 0.05

                    # CompLikelihood[i] = 0
    #parameter=sampler(1)[0]
    #vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    a.DLcon=a.DLcon
    a.DLx=a.DLx
    a.DLy=a.DLy
    a.x0=a.x0
    a.y0=a.y0
    
    # #print a.xm   
    #N=1000
# field data likelihood 
#    relt =IniLikelihood(a,sampler(1)[0])
# prior data likelihood
    #relt =prior(a,sampler(1)[0])
    
# integrate 
    a.x=x
    a.y=y
    t=a.t

#depth caculation

    #print a
    # N = 100000

    N = 100
    prob = Likelihood(a,N)[0]
    parameter = sampler(N)
    a.r=parameter[np.argmax(prob)]
    a.MaxLogLike = Likelihood(a,N)[1]    
    progressBar.setValue(50)
#________________confidence bounds____________
    if ourinformation['Method'] == 'Minimum':  
        bounds=[(-3.0,3.0),(-3.0,3.0),(-3.0,3.0),(-3.0,3.0)]    
        result = differential_evolution(partial(IniLikelihood1,a),bounds)  
        print "result=",result 
    #    result = optimize.minimize(partial(IniLikelihood1,a), initial_guess,method = 'SLSQP',bounds=[])#method=)#method='Powell')#method = 'SLSQP')#)#method='COBYLA')#)method='nelder-mead')#
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
    #     print a.fitted_params#,Dx1#,Dx2#,Dx3

        bounds=[(-3.0,3.0),(-3.0,3.0),(-3.0,3.0),(-3.0,3.0)]    
        result2 = differential_evolution(partial(IniLikelihood2,a),bounds) 
        print "result2=", result2
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
        bounds=[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)]    
        result3 = differential_evolution(partial(IniLikelihood3,a),bounds)
        print "result3=",result3     
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
        bounds=[(0.01,0.89),(0.01,0.89),(0.01,0.89),(0.01,0.89)]    
        result4 = differential_evolution(partial(IniLikelihood4,a),bounds)
        print "result4=",result4
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
        print "result5=",result5    
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

        gamma1 = a.r[20]
        gamma2 = a.r[21]
        gamma3 =a.r[22]
        gamma4 = a.r[23]
    #    a.par = (-0.15360589366043342, 2.6069143558650794, -1.8448758519500417, 2.2625958634456476, -0.27491226138924263, 0.21040698156590887, -1.5607634916644499, 0.63606729289101804, 0.8889381258608573, 0.73422840889420971, 0.10290151198336506, 0.33324096909555417, 0.88116098620896599, 0.64747290716440409, 0.81413080438311991, 0.87879107977393978, 0.51768264095519112, 0.092170034760164332, 0.60351934242547034, -0.13147021918728802, 0.35421863636294088, 0.22574843594175353, 0.16605840211125283, 0.25397452558405259)
        a.par = vx1,vx2,vx3,vx4,vy1,vy2,vy3,vy4,Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4
   
        rescf = []
        for u in range(len(t)):
            a.t = t[u]
            a.ti = a.Ti[u]
            resacf = []
            loc = zip(x,y)
            resacf=np.array(multicore3(a,loc))
            resacf=np.transpose(resacf)
            # sumcf = 0
            # for i in resacf:
            #     sumcf = sumcf +np.array(i)
            rescf.append(resacf)
            print rescf

        scf = np.array(rescf)
        scf = scf/np.max(scf)

    elif ourinformation['Method'] == 'Best':
        pass 
    #_______________confidence bounds_____________

    res = []
    zmin = []
    zmax = []
    #print t 
   # print a.Ti 
   # print zip(x,y)
   ####depth caculation

    if ourinformation['contour'] == 'contour':
        Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.xNode+1)
        Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.yNode+1)     
        [xd,yd] = np.meshgrid(Xa,Ya)
        xd = np.concatenate(xd)
        yd = np.concatenate(yd)
        locs = zip(xd,yd)
        z = np.array(multicore4(a,locs))
        zmax = z[:,0]
        zmin = z[:,1]
#-------------------------pending----
    de = oceansdb.ETOPO()        
    bat = []
    print "minbathy=",a.bathymin
    for i in range(len(xxa)):
        d = de['topography'].extract(lat=xxa[i], lon=yya[i])
        bathy = d['height']
        bath = abs(bathy)
        if bath < a.bathymin: #or bath > a.bathymax:
            bath = 1 
        else:
            bath = bath       
        bat.append(bath)
    bat = np.array(bat)
    #bat = np.transpose(bat)

    plt.contourf(Yya,Xxa,bat.reshape(len(Yya),len(Xxa)))
    plt.colorbar()
    time = datetime.datetime.now()
    time = time.strftime("%Y-%m-%d %H-%M-%S")
    fffilename  = "Results/bathy"+time+"_back.png"
    plt.savefig(fffilename)
    plt.clf()
############end###################
    for u in range(len(t)):
        a.t = t[u]
        a.ti = a.Ti[u]
        resa = []
        loc = zip(x,y)
        resa = np.array(multicore2(a,loc))
        #resa = np.transpose(resa)
        sum = 0
        for i in range(len(bat)):
            if bat[i] == 1: 
                resa[i] = 0   
            else:
                pass
            res.append(resa)
    res = np.array(res)
    s = res/np.max(res) 
    progressBar.setValue(75)
    print "s=",s 
    xkm = Xa 
    ykm = Ya
    xaxis = []
    newx = np.ones([len(xkm)])
    for i in range(len(xkm)):
        newx[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[i],ykm[0])).km
        xaxis.append(newx[i])

    yaxis = []
    newy = np.ones([len(xkm)])
    for i in range(len(ykm)):
        newy[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],ykm[i])).km
        yaxis.append(newy[i])

    #km field data
    SXkm = []
    newxfield = np.ones([len(a.DLlat)])
    for i in range(len(a.DLlon)):
        newxfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],a.DLlon[i])).km 
        SXkm.append(newxfield[i])
    SYkm = []
    newyfield = np.ones([len(a.DLlat)])
    for i in range(len(a.DLlon)):
        newyfield[i] = geopy.distance.distance((xkm[0],ykm[0]),(a.DLlat[i],ykm[0])).km 
        SYkm.append(newyfield[i])

    lon0km = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],lon0)).km 
    lat0km = geopy.distance.distance((xkm[0],ykm[0]),(lat0,ykm[0])).km 

# plot in km 
    for tt in range(len(t)):
        plt.figure()
        plt.rcParams['font.size'] = 13   # change the font size of colorbar
        plt.rcParams['font.weight'] = 'bold' # make the test bolder
        if ourinformation['Method'] == 'Best':
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)))
                plt.colorbar()                 
                if ourinformation['Plot'] =="nofield":
                    pass
                elif ourinformation['Plot']=="field":
                    plt.scatter(a.DLlon,a.DLlat,s=a.DLcon)                                      
                if ourinformation['contour'] =="nocontour":
                    pass                    
                elif ourinformation['contour'] == 'contour':
                    if all_same(zmin) == True:
                        print "All the upper depth is equal to", zmin[0]
                    elif all_same(zmin) == False: 
                        cs = plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)),4)
                        manual_locations = [(-88.7,28.4),(-88.6,28.6),(-88.3,28.5),(-88.1,28.4)]
                        plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    if all_same(zmax) == True:
                        print "All the lower depth is equal to", zmax[0]
                    elif all_same(zmax) == False:    
                        cs2 = plt.contour(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
                        manul = [(-88.0,28.5)]
                        plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)
                         
            elif ourinformation['Map'] == 'km': 
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(Xa),len(Ya)))
                plt.colorbar()
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=a.DLcon)
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":                    
                    if all_same(zmin) == True:
                        print "All the upper depth is equal to", zmin[0]
                    elif all_same(zmin) == False: 
                        cs = plt.contour(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya)),4)
                        #manual_locations = [(3,3),(5,15),(15,5),(20,15)]
                        plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)
                    if all_same(zmax) == True:    
                        print "All the lower depth is equal to", zmax[0]                       
                    elif all_same(zmax) == False:     
                        cs2 = plt.contour(xaxis,yaxis,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')   
                        #manul = [(20,10)]    
                        plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)  

        elif ourinformation['Method'] == 'Minimum':                
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)))
                plt.colorbar()                   
                plt.contour(Ya,Xa,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g'])    
      ######add the minimum regret part           
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.DLlon,a.DLlat,s=a.DLcon)
                elif ourinformation['Plot'] =="nofield":  
                    pass
                if ourinformation['contour'] =='contour':
                    if all_same(zmin) == True:
                        print "All the upper depth is equal to", zmin[0]
                    elif all_same(zmin) == False:                         
                        cs = plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)),4)
                        manual_locations = [(-88.7,28.4),(-88.6,28.6),(-88.3,28.5),(-88.1,28.4)]
                        plt.clabel(cs1,incline=1,fontsize=10,weight='bold')#,manual=manual_locations) 
                    if all_same(zmax) == True:
                        print "All the lower depth is equal to", zmax[0]  
                    elif all_same(zmax) == False:   
                        cs2 = plt.contour(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
                        manul = [(-88.0,28.5)]                    
                        plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)                            
                elif ourinformation['contour'] =="nocontour":
                    pass
            elif ourinformation['Map'] == 'km':
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(Xa),len(Ya)))
                plt.colorbar()                
                plt.contour(xaxis,yaxis,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g']) 
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=a.DLcon)
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":
                    if all_same(zmin) == True:
                        print "All the upper depth is equal to", zmin[0]
                    elif all_same(zmin) == False:                                               
                        cs = plt.contour(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya)),4)
                        #manual_locations = [(3,3),(5,15),(15,5),(20,15)]
                        plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)
                    if all_same(zmax) == True: 
                        print "All the lower depth is equal to", zmax[0]  
                    elif all_same(zmax) == False:                         
                        cs2 = plt.contour(xaxis,yaxis,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
                        #manul = [(20,10)]
                        plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)  

        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        fffilename  = "Results/submerged"+time+"_back.png"
        plt.savefig(fffilename)
        plt.clf()
        img = cv2.imread(fffilename)
        crop_img = img[40:440,500:580]
        # cv2.imshow("image",crop_img)
        fffilename1 = "Results/submerged"+time+"_legend.png"
        cv2.imwrite(fffilename1,crop_img)
        plt.clf()

###############add the km display##########
	if ourinformation['Map'] == 'Coordinate':    
	    	if ourinformation['contour'] =="contour":
		        if all_same(zmin) == True:
		            print "All the upper depth is equal to", zmin[0]
		        elif all_same(zmin) == False:
		            plt.figure()		            
		            cs1 = plt.contourf(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya))) 			        		          
		            plt.clabel(cs1,incline=1,fontsize=10,weight='bold')        
		            plt.colorbar(cs1,shrink=1.0)         
		            contourname = "Results/submerged"+time+"_Isopycnalupper.png"
		            plt.savefig(contourname)
		        if all_same(zmax) == True:
		            print "All the lower depth is equal to", zmax[0] 
		        elif all_same(zmax) == False:    
		            plt.figure()         
		            cs2 = plt.contourf(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)))
		            plt.clabel(cs2,incline=1,fontsize=10,weight='bold')        
		            plt.colorbar(cs2,shrink=1.0)         
		            contour = "Results/submerged"+time+"_Isopycnallower.png"
		            plt.savefig(contour)
		        elif ourinformation['contour'] =="nocontour":
		        	pass
	elif ourinformation['Map'] == 'km':
	    	if ourinformation['contour'] =="contour":
		        if all_same(zmin) == True:
		            print "All the upper depth is equal to", zmin[0]
		        elif all_same(zmin) == False:
		            plt.figure()		            			        
		            cs1 = plt.contourf(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya))) 		          
		            plt.clabel(cs1,incline=1,fontsize=10,weight='bold')        
		            plt.colorbar(cs1,shrink=1.0)         
		            contourname = "Results/submerged"+time+"_Isopycnalupper.png"
		            plt.savefig(contourname)
		        if all_same(zmax) == True:
		            print "All the lower depth is equal to", zmax[0] 
		        elif all_same(zmax) == False:    
		            plt.figure()         
		            cs2 = plt.contourf(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya)))  
		            plt.clabel(cs2,incline=1,fontsize=10,weight='bold')        
		            plt.colorbar(cs2,shrink=1.0)         
		            contour = "Results/submerged"+time+"_Isopycnallower.png"
		            plt.savefig(contour)
		        elif ourinformation['contour'] =="nocontour":
		        	print "no contour"	    	
	plt.clf()    	
#########################showing figure###########
	print "_______________________________Above is True_________________________________________"
	
	if ourinformation['Method'] == 'Best':                     
	        if ourinformation['Map'] == 'Coordinate':    
	            plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)))
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
	                plt.scatter(a.DLlon,a.DLlat,s=a.DLcon)
	            elif ourinformation['Plot'] == 'nofield':
	                pass
	            if ourinformation['contour'] =='contour':
	                if all_same(zmin) == True:
	                    print "All the upper depth is equal to", zmin[0]
	                elif all_same(zmin) == False: 
	                    #print "zzzz=1"                    
	                    plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)),4)
	                    cs = plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)),4)
	                    #manual_locations = [(-88.7,28.4),(-88.6,28.6),(-88.3,28.5),(-88.1,28.4)]
	                    plt.clabel(cs,incline=1,fontsize=6,weight='bold')#,manual=manual_locations) 
	                if all_same(zmax) == True:
	                    print "All the lower depth is equal to", zmax[0]  
	                elif all_same(zmax) == False:   
	                    cs2 = plt.contour(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
	                    #manul = [(-88.0,28.5)]                    
	                    plt.clabel(cs2,incline=1,fontsize=6,weight='bold')#,manual=manul)                          
	            elif ourinformation['contour'] =='nocontour':
	                pass

	        if ourinformation['Map'] == 'km': 
	            plt.contourf(xaxis,yaxis,s[tt].reshape(len(xa),len(ya)))
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
	                plt.scatter(SXkm,SYkm,s=a.DLcon,c="r",label="Field Data")#,marker=r'$\clubsuit$')
	                print "I am field"             
	            elif ourinformation['Plot'] == 'nofield':
	                print "nofield"
	            if ourinformation['contour'] =='contour':
	                if all_same(zmin) == True:
	                    print "All the upper depth is equal to", zmin[0]
	                elif all_same(zmin) == False:                                               
	                    cs = plt.contour(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya)),4)
	                    #manual_locations = [(3,3),(5,15),(15,5),(20,15)]
	                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)
	                if all_same(zmax) == True: 
	                    print "All the lower depth is equal to", zmax[0]  
	                elif all_same(zmax) == False:                         
	                    cs2 = plt.contour(xaxis,yaxis,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
	                    #manul = [(20,10)]
	                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul) 

	elif ourinformation['Method'] == 'Minimum':
	    if ourinformation['Map'] == 'Coordinate':    #### add minimum part 
	        plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)))
                plt.contour(Ya,Xa,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g'])	            
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
	                plt.scatter(a.DLlon,a.DLlat,s=a.DLcon)
                elif ourinformation['Plot'] == 'nofield':
	                print "nofield"
                if ourinformation['contour'] =='contour':
	                if all_same(zmin) == True:
	                    print "All the upper depth is equal to", zmin[0]
	                elif all_same(zmin) == False:                         
	                    cs = plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)),4)
	                    #manual_locations = [(-88.7,28.4),(-88.6,28.6),(-88.3,28.5),(-88.1,28.4)]
	                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations) 
	                if all_same(zmax) == True:
	                    print "All the lower depth is equal to", zmax[0]  
	                elif all_same(zmax) == False:   
	                    cs2 = plt.contour(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
	                    manul = [(-88.0,28.5)]                    
	                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)    
                elif ourinformation['contour'] =='nocontour':
	                pass

	    elif ourinformation['Map'] == 'km': 
	        print "I am km"
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(xa),len(ya)))
                plt.contour(xaxis,yaxis,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g'])           
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
	                plt.scatter(SXkm,SYkm,s=a.DLcon,c="r",label="Field Data")#,marker=r'$\clubsuit$')
	                print "I am field"             
                elif ourinformation['Plot'] == 'nofield':
	                print "nofield"
                if ourinformation['contour'] =='contour':
	                if all_same(zmin) == True:
	                    print "All the upper depth is equal to", zmin[0]
	                elif all_same(zmin) == False:                                               
	                    cs = plt.contour(xaxis,yaxis,np.array(zmin).reshape(len(xa),len(ya)),4)
	                    manual_locations = [(3,3),(5,15),(15,5),(20,15)]
	                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)
	                if all_same(zmax) == True: 
	                    print "All the lower depth is equal to", zmax[0]  
	                elif all_same(zmax) == False:                         
	                    cs2 = plt.contour(xaxis,yaxis,np.array(zmax).reshape(len(xa),len(ya)),1,colors='r')                          
	                    manul = [(20,10)]
	                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul)         

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

        # plt.figure()
        # plt.contourf(Ya,Xa,prob[tt].reshape(len(xa),len(ya))) 
        # plt.colorbar()       
        # cs1 = plt.contour(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)))  
        # plt.clabel(cs1,incline=1,fontsize=7)
        # plt.title('lowest contour')        
        # plt.grid()
        # plt.plot(-88.366,28.738,'ro',ms=3)


#     plt.figure()
#     cs = plt.contourf(Ya,Xa,np.array(zmax).reshape(len(xa),len(ya)))  
#     plt.colorbar(cs,shrink=1.0)
# #    plt.clabel(cs,incline=1,fontsize=7)
#     plt.title('deepest contour')
      
#     plt.figure()
#     cs1 = plt.contourf(Ya,Xa,np.array(zmin).reshape(len(xa),len(ya)))  
#     #plt.clabel(cs1,incline=1,fontsize=7)
#     plt.title('lowest contour')         
#     plt.colorbar(cs1,shrink=1.0) 
 #   print MaxLogLike
    # plt.show()
