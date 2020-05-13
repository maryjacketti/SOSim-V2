# -*- coding: utf-8 -*-
from __future__ import division
import itertools
from itertools import groupby
from math import *
import numpy as np
import random
import utm
import matplotlib.pyplot as plt
import matplotlib.ticker
from functools import partial
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import integrate
from  scipy.optimize import differential_evolution
from  scipy.optimize import fmin_tnc
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
import geopy.distance
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xlsxwriter
from sklearn.cluster import KMeans
import cv2
import xarray
from scipy.interpolate import griddata
#import math as ma

ourinformation = {}

def ff(x,y,mux,muy,Dx,Dy,ro):
    s=1./(2*np.pi*Dx*Dy*np.sqrt(1-ro**2))*np.exp(-1/(2*(1-ro**2))*((x-mux)**2/Dx**2+(y-muy)**2/Dy**2-2*ro*(x-mux)*(y-muy)/(Dx*Dy)))#stats.norm(loc=mu,scale=Dx).pdf(x)
    return s    

def fcont(x,y,vx,vy,Dx,Dy,ro,x0,y0,t,s,sigmax0,sigmay0):
    k=0
    for ii in range(len(s)):
        mux = x0[ii] + vx*(t-s[ii])
        muy = y0[ii] + vy*(t-s[ii])
        sx = np.sqrt(2.0*Dx*(t-s[ii]))+sigmax0[ii]
        sy = np.sqrt(2.0*Dy*(t-s[ii]))+sigmay0[ii]
        mm = ff(x,y,mux,muy,sx,sy,ro)
        k = k + mm/len(s)#1/len(s)*ff(x,y,mux,muy,sx,sy,ro[i])
    return k

def CalTime(a,b):
    start = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    ends = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    diff = ends - start
    return diff.total_seconds()/86400.

def sampler(N):
    #print ourinformation
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
    return zip(vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4)

def sampleprior(a): # sample data from OSCAR 
    rdata = []
    da = []
    uni = []
    DLxx = []
    DLyy = []
    DLccon = []
    priortt = []
    YY = []
    ccc = []
    ee = []
    priorx = []
    priory = []
    xx = int(ourinformation["Ratio"])#a.N*repeat
    N = 15*(1./xx)
    print 'N',N 
    n = ceil(np.sqrt(N))-1
    a.size = 3#a.size 
    scale = [0.2,0.2]
    a.X = np.linspace((a.lat0)-scale[0]-0.005,(a.lat0)+scale[0]+0.005,n+1)
    a.Y = np.linspace((a.lon0)-scale[1]-0.005,(a.lon0+0.005)+scale[1],n+1) 
    a.arr=[[[] for arx in range(len(a.X)-1)] for ary in range(len(a.Y)-1)]  

    for p in range(len(a.DLclp)):
        if a.priorlon[p]>=a.Y[0] and a.priorlon[p]<=a.Y[-1] and a.priorlat[p]>=a.X[0] and a.priorlat[p]<=a.X[-1] and a.ttp[p]==a.priorsampletime:#
            l=int((a.priorlat[p]-a.X[0])/(2*(scale[0]+0.005)/n))
            m=int((a.priorlon[p]-a.Y[0])/(2*(scale[1]+0.005)/n))
            a.arr[l][m].append((a.priorlat[p],a.priorlon[p],a.DLclp[p],a.ttp[p]))  # change later    

    for l in range(len(a.X)-1):
        for m in range(len(a.Y)-1):
            if len(a.arr[l][m]) <= a.size:
                rdata.append(a.arr[l][m])
            else:
                np.random.seed(0)
                ind=np.random.choice(range(len(a.arr[l][m])),a.size,replace=True)
                tmp = np.array(a.arr[l][m])
                rdata.append(tmp[ind])
   
    for g in range(len(rdata)):
        for o in range(len(rdata[g])):
            da.append(rdata[g][o])
    rdata = np.array(da)
    lat = rdata[:,0]
    lon = rdata[:,1]
    con = rdata[:,2]
    priort = rdata[:,3]
    camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(lat,lon)])
    priorx.append(np.array(map(float,camdatalist[:,0]))/1000)  #381
    priory.append(np.array(map(float,camdatalist[:,1]))/1000)  #3170  
    return lat,lon,con,priorx[0],priory[0],priort

def IniLikelihoodcal(a,parameter):
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    DLx = a.priorxx
    DLy = a.prioryy
    DLcon = a.priorccon
    st = a.st
    s=a.STT
    x01 = a.xx01
    y01 = a.yy01 
    x02 = a.xx02 
    y02 = a.yy02 
    x03 = a.xx03 
    y03 = a.yy03
    x04 = a.xx04 
    y04 = a.yy04    
    sigmax01 = a.ssigmax01 
    sigmay01 = a.ssigmay01                
    sigmax02 = a.ssigmax02 
    sigmay02 = a.ssigmay02 
    sigmax03 = a.ssigmax03 
    sigmay03 = a.ssigmay03                
    sigmax04 = a.ssigmax04 
    sigmay04 = a.ssigmay04      
    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0#np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=np.ones([len(DLx)])

    for i in range(len(DLx)):
        Prob = gamma1*fcont(DLx[i],DLy[i],vx1,vy1,Dx1,Dy1,ro1,x01,y01,st,s,sigmax01,sigmay01)\
        + gamma2*fcont(DLx[i],DLy[i],vx2,vy2,Dx2,Dy2,ro2,x02,y02,st,s,sigmax02,sigmay02)  \
        +gamma3*fcont(DLx[i],DLy[i],vx3,vy3,Dx3,Dy3,ro3,x03,y03,st,s,sigmax03,sigmay03)  +\
        gamma4*fcont(DLx[i],DLy[i],vx4,vy4,Dx4,Dy4,ro4,x04,y04,st,s,sigmax04,sigmay04)  
        if Prob>1e-308:
            Lamda = 1/Prob                    
            IniIndLikelihood[i] = np.log(Lamda)-Lamda*DLcon[i]#*(1.0/c)

        else:
            IniIndLikelihood[i] = 0
            Lamda = 0

    for ci in range(len(DLx)):
        if DLcon[ci]>0:
            if IniIndLikelihood[ci] == 0:
                CompLikelihood[i] = 0
    return IniIndLikelihood

def IniLikelihoodcalfield(a,parameter):
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    DLx = a.Datax
    DLy = a.Datay
    DLcon = a.Datacon 
    st = a.st
    s=a.STT    
    x01 = a.xx01
    y01 = a.yy01 
    x02 = a.xx02 
    y02 = a.yy02 
    x03 = a.xx03 
    y03 = a.yy03
    x04 = a.xx04 
    y04 = a.yy04   
    sigmax01 = a.ssigmax01 
    sigmay01 = a.ssigmay01                
    sigmax02 = a.ssigmax02 
    sigmay02 = a.ssigmay02 
    sigmax03 = a.ssigmax03 
    sigmay03 = a.ssigmay03                
    sigmax04 = a.ssigmax04 
    sigmay04 = a.ssigmay04

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=0#np.ones([len(DLx)])

    for i in range(len(DLx)):
        Prob = gamma1*fcont(DLx[i],DLy[i],vx1,vy1,Dx1,Dy1,ro1,x01,y01,st,s,sigmax01,sigmay01)\
        + gamma2*fcont(DLx[i],DLy[i],vx2,vy2,Dx2,Dy2,ro2,x02,y02,st,s,sigmax02,sigmay02)  \
        +gamma3*fcont(DLx[i],DLy[i],vx3,vy3,Dx3,Dy3,ro3,x03,y03,st,s,sigmax03,sigmay03)  +\
        gamma4*fcont(DLx[i],DLy[i],vx4,vy4,Dx4,Dy4,ro4,x04,y04,st,s,sigmax04,sigmay04)  
        if Prob>1e-308:
            Lamda = 1/Prob                    
            IniIndLikelihood[i] = np.log(Lamda)-Lamda*DLcon[i]#*(1.0/c)

        else:
            IniIndLikelihood[i] = 0
            Lamda = 0

    return IniIndLikelihood

def priorlikelihood(a,N):
    parameter = sampler(N)     
    Inilike = np.array(multicore7(parameter,a))
    IniIndLikelihood = np.transpose(Inilike)
    DLx = a.priorxx
    DLcon = a.priorccon
    CompLikelihood =np.ones(N)
    FileLikelihood = np.zeros(N)
    Likelihoodi = np.zeros(N)  
    l = a.l
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
                    FileLikelihood[i] =  FileLikelihood[i] + IniIndLikelihood[ci,i]
        if CompLikelihood[i]==1:          
            if MaxLogLike ==-22:
                MaxLogLike=FileLikelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,FileLikelihood[i]])
    MaxLogLike = MaxLogLike/l
    for i in range(N):
        if CompLikelihood[i]==1:            
            FileLikelihood[i] = FileLikelihood[i]/l - MaxLogLike + 7            
    return FileLikelihood,MaxLogLike

def fieldpriorlikelihood(a,N):  
    parameter = sampler(N)     
    Inilike = np.array(multicore8(parameter,a))    
    IniIndLikelihood = np.transpose(Inilike)
    Inilikeprior = np.array(multicore7(parameter,a))
    InipriorLikelihood = np.transpose(Inilikeprior) 
    InipriorLikelihood = InipriorLikelihood * (1/a.l) 
    Ini = np.concatenate((IniIndLikelihood, InipriorLikelihood), axis=0)

    CompLikelihood =np.ones(N)    
    FileLikelihood = np.zeros(N)
    Likelihoodi = np.zeros(N)  
    DLx = a.Datax
    DLcon = a.Datacon
    DLxx = a.priorxx
    DLccon = a.priorccon
    xx = np.concatenate((DLx, DLxx), axis=None)
    con = np.concatenate((DLcon, DLccon), axis=None)
    for i in range(N):
        for ci in range(len(xx)):
            if con[ci]>0:
                if Ini[ci,i] == 0:
                    CompLikelihood[i] = 0
    
    MaxLogLike = -22
    for i in range(N):
        for ci in range(len(xx)):
            if con[ci]>0:
                if CompLikelihood[i]==1:
                    FileLikelihood[i] = FileLikelihood[i] + Ini[ci,i] 

        if CompLikelihood[i]==1:          
            if MaxLogLike ==-22:
                MaxLogLike=FileLikelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,FileLikelihood[i]])    

    for i in range(N):
        if CompLikelihood[i]==1:
            FileLikelihood[i] = FileLikelihood[i] - MaxLogLike + 7
            Likelihoodi[i] = np.exp(FileLikelihood[i])
    loc = np.argmax(FileLikelihood)
    print 'loc',loc
    return FileLikelihood,MaxLogLike

def fieldlikelihood(a,N):
    parameter = sampler(N)     
    Inilike = np.array(multicore8(parameter,a))    
    IniIndLikelihood = np.transpose(Inilike)
    DLx = a.Datax
    CompLikelihood =np.ones(N)
    DLcon = a.Datacon
    CompLikelihood =np.ones(N)
    FileLikelihood = np.zeros(N)
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
                if CompLikelihood[i] == 1:
                    FileLikelihood[i] =  FileLikelihood[i] + IniIndLikelihood[ci,i]
        if CompLikelihood[i]== 1:          
            if MaxLogLike == -22:
                MaxLogLike=FileLikelihood[i]
            else:
                MaxLogLike=np.max([MaxLogLike,FileLikelihood[i]])
    MaxLogLike = MaxLogLike
    for i in range(N):
        if CompLikelihood[i]==1:            
            FileLikelihood[i] = FileLikelihood[i] - MaxLogLike + 7            
    return FileLikelihood,MaxLogLike
    
def IniLikelihood(a,parameter):
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4=parameter
    DLx = a.DLx
    print 'DLx',DLx
    DLy = a.DLy
    DLcon = a.DLcon 
    st = a.st  
    if a.ourinformation['Type'] == 'instantaneous':
        s = [0]
    if a.ourinformation['Type'] == 'continuous': 
        s = a.TTi  
    x01 = a.xx01
    y01 = a.yy01 
    x02 = a.xx02 
    y02 = a.yy02 
    x03 = a.xx03 
    y03 = a.yy03
    x04 = a.xx04 
    y04 = a.yy04   
    sigmax01 = a.ssigmax01 
    sigmay01 = a.ssigmay01                
    sigmax02 = a.ssigmax02 
    sigmay02 = a.ssigmay02 
    sigmax03 = a.ssigmax03 
    sigmay03 = a.ssigmay03                
    sigmax04 = a.ssigmax04 
    sigmay04 = a.ssigmay04
    print 'lenx01',len(x01)
    print 'len(s)',len(s)
    print 'lensigmax',len(a.ssigmax01)

    IniIndLikelihood = np.ones([len(DLx)])
    Lamda = 0# np.zeros([len(vx1)])
    Prob = 0#np.zeros([len(vx1)])
    CompLikelihood=1##np.ones([len(vx1)])

    xx = []
    pprob = []

    for i in range(len(DLx)):
        x = DLx[i]
        y = DLy[i]
        prob = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,x01,y01,st,s,sigmax01,sigmay01)\
        + gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,x02,y02,st,s,sigmax02,sigmay02)  \
        +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,x03,y03,st,s,sigmax03,sigmay03)  \
        +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,x04,y04,st,s,sigmax04,sigmay04)          
        xx.append(x) 
        pprob.append(prob)

    xx = np.array(xx)
    pprob = np.array(pprob)

    for i in range(len(DLx)):
        if (DLx[i] in xx) == False: 
                Prob = gamma1*fcont(DLx[i],DLy[i],vx1,vy1,Dx1,Dy1,ro1,x01,y01,st,s,sigmax01,sigmay01)\
                + gamma2*fcont(DLx[i],DLy[i],vx2,vy2,Dx2,Dy2,ro2,x02,y02,st,s,sigmax02,sigmay02)  \
                +gamma3*fcont(DLx[i],DLy[i],vx3,vy3,Dx3,Dy3,ro3,x03,y03,st,s,sigmax03,sigmay03)  +\
                gamma4*fcont(DLx[i],DLy[i],vx4,vy4,Dx4,Dy4,ro4,x04,y04,st,s,sigmax04,sigmay04)  
                if Prob>1e-308:
                    Lamda = 1/Prob                    
                    IniIndLikelihood[i] = np.log(Lamda)-Lamda*DLcon[i]#*(1.0/c)

                else:
                    IniIndLikelihood[i] = 0
                    Lamda = 0
        if (DLx[i] in xx) == True:
            ccc = np.where(DLx[i]==xx)
            Prob = pprob[ccc[0]]
            if Prob>1e-308:
                    Lamda = 1/Prob                    
                    #c = np.exp((100-Lamda)/b)/(1*np.exp((100-Lamda)/b))-np.exp((-Lamda)/b)/(1+np.exp((-Lamda)/b))
                    #if c > 1e-308:
                    IniIndLikelihood[i] = np.log(Lamda)-Lamda*DLcon[i]#*(1.0/c)
            else:
                    IniIndLikelihood[i] = 0
                    Lamda = 0

    for ci in range(len(DLx)):
        if DLcon[ci]>0:
            if IniIndLikelihood[ci] == 0:
                CompLikelihood = 0
    return IniIndLikelihood

def integ(a,loc):
    parameter = a.ttt
    ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter[16:24]
    [x,y]=loc  
    t=a.t
    mx1=a.xx01
    my1=a.yy01
    mx2=a.xx02
    my2=a.yy02
    mx3=a.xx03
    my3=a.yy03
    mx4=a.xx04
    my4=a.yy04                
    Dx1=a.Dx01
    Dy1=a.Dy01
    Dx2=a.Dx02
    Dy2=a.Dy02        
    Dx3=a.Dx03
    Dy3=a.Dy03
    Dx4=a.Dx04
    Dy4=a.Dy04           
    ProObsGivenPar = gamma1*ff(x,y,mx1,my1,Dx1,Dy1,ro1) \
    +gamma2*ff(x,y,mx2,my2,Dx2,Dy2,ro2) \
    +gamma3*ff(x,y,mx3,my3,Dx3,Dy3,ro3) \
    +gamma4*ff(x,y,mx4,my4,Dx4,Dy4,ro4)
    return ProObsGivenPar

def integcontinuous(a,loc):
    parameter = a.r
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter
    [x,y]=loc  
    t=a.Pt
    x01 = a.xx1
    x02 = a.xx2
    x03 = a.xx3
    x04 = a.xx4
    y01 = a.yy1
    y02 = a.yy2
    y03 = a.yy3
    y04 = a.yy4
    sx1 = a.sx1
    sy1 = a.sy1    
    sx2 = a.sx2
    sy2 = a.sy2 
    sx3 = a.sx3
    sy3 = a.sy3    
    sx4 = a.sx4
    sy4 = a.sy4
    s = a.ps       
    ProObsGivenPar = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,x01,y01,t,s,sx1,sy1) \
    +gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,x02,y02,t,s,sx2,sy2) \
    +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,x03,y03,t,s,sx3,sy3) \
    +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,x04,y04,t,s,sx4,sy4)
    return ProObsGivenPar

def integcfconti(a,loc):
    parameter = a.r
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter  
    [x,y]=loc
    t=a.Pt
    mx1=a.xx1
    my1=a.yy1
    mx2=a.xx2
    my2=a.yy2
    mx3=a.xx3
    my3=a.yy3
    mx4=a.xx4
    my4=a.yy4                
    sx1 = a.sx1
    sy1 = a.sy1    
    sx2 = a.sx2
    sy2 = a.sy2 
    sx3 = a.sx3
    sy3 = a.sy3    
    sx4 = a.sx4
    sy4 = a.sy4 
    s = a.ps  
    ProObsGivenPar = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,mx1,my1,t,s,sx1,sy1) \
    +gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,mx2,my2,t,s,sx2,sy2) \
    +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,mx3,my3,t,s,sx3,sy3) \
    +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,mx4,my4,t,s,sx4,sy4)
    return ProObsGivenPar

def integcfcontiins(a,loc):
    parameter = a.r
    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = parameter  
    [x,y]=loc
    t=a.Pt
    mx1=[a.xx01]
    my1=[a.yy01]
    mx2=[a.xx02]
    my2=[a.yy02]
    mx3=[a.xx03]
    my3=[a.yy03]
    mx4=[a.xx04]
    my4=[a.yy04]               
    sx1 = [a.Dx01]
    sy1 = [a.Dy01]    
    sx2 = [a.Dx02]
    sy2 = [a.Dy02] 
    sx3 = [a.Dx03]
    sy3 = [a.Dy03]    
    sx4 = [a.Dx04]
    sy4 = [a.Dy04] 
    s = [0]  
    print 'mx1',mx1
    print len(mx1)
    ProObsGivenPar = gamma1*fcont(x,y,vx1,vy1,Dx1,Dy1,ro1,mx1,my1,t,s,sx1,sy1) \
    +gamma2*fcont(x,y,vx2,vy2,Dx2,Dy2,ro2,mx2,my2,t,s,sx2,sy2) \
    +gamma3*fcont(x,y,vx3,vy3,Dx3,Dy3,ro3,mx3,my3,t,s,sx3,sy3) \
    +gamma4*fcont(x,y,vx4,vy4,Dx4,Dy4,ro4,mx4,my4,t,s,sx4,sy4)
    return ProObsGivenPar

def depthcalculate(a,locs):
    delta = 25  # default 
    de = oceansdb.ETOPO()
    [xd,yd] = locs
    db = oceansdb.CARS()
    d = de['topography'].extract(lat=xd, lon=yd)
    bathy = d['height']
    bath = abs(bathy)
    if bath < 800:
        depthmax = 0
        depthmin = 0         
    if bath <= 1300: 
        depthas = np.arange(800,bath,delta)  # change the depth later   
    if bath >1300:  
        depthas = np.arange(800,1300,delta)  # change the depth later       
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
            densas.append(k)
    if len(densas) ==0:
        depthmax = 0
        depthmin = 0 
    else:
        maxden = np.max(densas)
        minden = np.min(densas)
        depthmax = depthas[maxden]
        depthmin = depthas[minden]
    return depthmax,depthmin

def all_same(items):
    return all(x==items[0] for x in items)

def Likelihood(a,N):
    print "This is Likelihood !"
    Result = []
    parameter = sampler(N)  
    #IniLikelihood=np.array(IniLikelihood(a,parameter[0]))       
    IniLikelihood=np.array(multicore1(a,parameter))
    IniIndLikelihood = np.transpose(IniLikelihood)
    DLx = a.DLx
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
    para = []
    vx1,vy1,vx2,vy2 = parameter    
    Max = a.MaxLogLike
    r = a.rr    
    print 'rr',r[4:24]
    vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = r[4:24]
    parameter = vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4     
    para = np.append(para,parameter)

    st = a.st
    s = a.STT
    l = a.l    
    totalsum = []
    if a.ourinformation["SubmergedType"] == 'Nodate':
        IniIndLikelihood = IniLikelihood(a,para)
    if a.ourinformation["SubmergedType"] == 'OSCAR':
        if len(a.priorxx) > 0:
            Ini = IniLikelihoodcal(a,para)
            Ini = np.array(Ini)
            Inisum = np.sum(Ini)
            Inisum = Inisum/l
            totalsum = np.append(totalsum,Inisum)
        #print 'Inisum',Inisum
        if len(a.Datax) > 0: 
            Inifield =IniLikelihoodcalfield(a,para)
            Inifieldsum = np.sum(Inifield)
            totalsum=np.append(totalsum,Inifieldsum)
        IniIndLikelihood =  totalsum

    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    return login 

def IniLikelihood2(a,parameter):
    para = []
    params = a.fitted_params
    vx1,vy1,vx2,vy2 = params
    vx3,vy3,vx4,vy4 = parameter    
    para = np.append(para,params)  
    para = np.append(para,parameter)  
    Max = a.MaxLogLike
    r = a.rr 
    Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = r[8:24]
    para = np.append(para,r[8:24])   
    st = a.st
    s=a.STT
    l = a.l    
    Inisum = []
    if a.ourinformation["SubmergedType"] == 'Nodate':
        IniIndLikelihood = IniLikelihood(a,para)
    if a.ourinformation["SubmergedType"] == 'OSCAR':
        Ini = IniLikelihoodcal(a,para)
        Ini = np.array(Ini)
        Inisum = np.sum(Ini)
        Inisum = Inisum/l            
        if len(a.Datax) > 0:         
            Inifield =IniLikelihoodcalfield(a,para)
            Inifieldsum = np.sum(Inifield)
            Inisum=np.append(Inisum,Inifieldsum)
        IniIndLikelihood =  Inisum

    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    return login 

def IniLikelihood3(a,parameter):
    para = []
    params = a.fitted_params
    vx1,vy1,vx2,vy2 = params
    para = np.append(para,params)
    params2 = a.fitted_params2
    vx3,vy3,vx4,vy4 = params2
    para = np.append(para,params2)      
    Dx1,Dy1,Dx2,Dy2 = parameter
    para = np.append(para,parameter)        
    Max = a.MaxLogLike
    r = a.rr    
    Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4=r[12:24]
    para = np.append(para,r[12:24])   
    st = a.st
    s=a.STT
    l = a.l    
    Inisum = []
    if a.ourinformation["SubmergedType"] == 'Nodate':
        IniIndLikelihood = IniLikelihood(a,para)
    if a.ourinformation["SubmergedType"] == 'OSCAR':
        Ini = IniLikelihoodcal(a,para)
        Ini = np.array(Ini)
        Inisum = np.sum(Ini)
        Inisum = Inisum/l            
        if len(a.Datax) > 0:         
            Inifield =IniLikelihoodcalfield(a,para)
            Inifieldsum = np.sum(Inifield)
            Inisum=np.append(Inisum,Inifieldsum)
        IniIndLikelihood =  Inisum

    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    #print login
    return login 

def IniLikelihood4(a,parameter):
    para = []
    params = a.fitted_params
    vx1,vy1,vx2,vy2 = params
    para = np.append(para,params)    
    params2 = a.fitted_params2
    vx3,vy3,vx4,vy4 = params2
    para = np.append(para,params2)
    params3 = a.fitted_params3
    Dx1,Dy1,Dx2,Dy2 = params3
    para = np.append(para,params3)
    Dx3,Dy3,Dx4,Dy4 = parameter       
    para = np.append(para,parameter)
    Max = a.MaxLogLike
    r = a.rr
    l = a.l    
    ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = r[16:24]
    para = np.append(para,r[16:24]) 
    Inisum = []   
    if a.ourinformation["SubmergedType"] == 'Nodate':
        IniIndLikelihood = IniLikelihood(a,para)
    if a.ourinformation["SubmergedType"] == 'OSCAR':
        Ini = IniLikelihoodcal(a,para)
        Ini = np.array(Ini)
        Inisum = np.sum(Ini)
        Inisum = Inisum/l         
        if len(a.Datax) > 0:         
            Inifield =IniLikelihoodcalfield(a,para)
            Inifieldsum = np.sum(Inifield)
            Inisum=np.append(Inisum,Inifieldsum)
        IniIndLikelihood =  Inisum
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    return login 

def IniLikelihood5(a,parameter):
    para = []
    Max = a.MaxLogLike
    r = a.rr    
    ro1,ro2,ro3,ro4 = parameter           
    params = a.fitted_params
    vx1,vy1,vx2,vy2 = params
    para = np.append(para,params)    
    params2 = a.fitted_params2
    vx3,vy3,vx4,vy4 = params2
    para = np.append(para,params2)    
    params3 = a.fitted_params3
    Dx1,Dy1,Dx2,Dy2 = params3
    para = np.append(para,params3)    
    params4 = a.fitted_params4
    Dx3,Dy3,Dx4,Dy4 = params4
    para = np.append(para,params4)
    para = np.append(para,parameter)  
    para = np.append(para,r[20:24])          
    l = a.l    
    Inisum = []       
    if a.ourinformation["SubmergedType"] == 'Nodate':
        IniIndLikelihood = IniLikelihood(a,para)
    if a.ourinformation["SubmergedType"] == 'OSCAR':
        Ini = IniLikelihoodcal(a,para)
        Ini = np.array(Ini)
        Inisum = np.sum(Ini)
        Inisum = Inisum/l         
        if len(a.Datax) > 0:         
            Inifield =IniLikelihoodcalfield(a,para)
            Inifieldsum = np.sum(Inifield)
            Inisum=np.append(Inisum,Inifieldsum)
        IniIndLikelihood =  Inisum
    d = chi2.ppf(float(a.ourinformation["confidence"]),25)
    login = abs(np.sum(IniIndLikelihood) - Max + d/2)
    return login 
#-----------------confidence bounds--------------------------
def plot(time,File,xa,ya):
    density = []
    ppy = []
    ds=[]
    for i in range(len(File)):
        ds.append(xarray.open_dataset(File[i]))
    data = xarray.merge(ds)
    pt = data.variables['MT'][:]
    print 'predicttime',data.variables['MT'][time]
    dep=data.variables['Depth'][:]
    px1 = data.variables['Longitude'][:]  # 1 dimension   
    px1 = np.array(px1)
    py1 = data.variables['Latitude'][:]
    py1 = np.array(py1)
    idx = (np.abs(px1 - ya[0])).argmin() # longitude 
    idx2 = (np.abs(px1 - ya[-1])).argmin() # longitude 
    idx3 = (np.abs(py1 - xa[0])).argmin() # latitude
    idx4 = (np.abs(py1 - xa[-1])).argmin()  # latitude 

    px = data.variables['Longitude'][idx-1:idx2+1]  # 1 dimension    
    upperlon = []
    upperlat = []
    upperdepth = []
    lowerlon = []
    lowerlat = []
    lowerdepth = []
    for i in range(idx3,idx4): # -27 is May 5 
        py = data.variables['Latitude'][i]
        py = np.array(py)
        salinity = data.variables['salinity'][time,:,i,idx-1:idx2+1] # first is time, second is the dept, third is latitude and last one is longitude
        temp = data.variables['temperature'][time,:,i,idx-1:idx2+1]
        temperatureas = T90conv(temp)
        densityas = sw.dens0(salinity,temperatureas)
        densityas = densityas - 1000
        [pxx,depx] = np.meshgrid(px,dep)
        #depx = -depx
        cs = plt.contour(pxx,depx,densityas)  # cs.collections is the number of the lines 
        b = cs.levels
        b = np.array(b)
        index = np.where((b>=27.65)&(b<=27.72))
        # the upper 
        if len(index[0])>1:
            cc = index[0][0]
            a = cs.collections[cc].get_paths()[0] # the points constract each line 
            v = a.vertices
            lon = v[:,0]
            depth = v[:,1]
            lat = np.repeat(py,len(lon))
            upperlon.append(lon)
            upperlat.append(lat)
            upperdepth.append(depth)
            # the lower depth 
            ccl = index[0][-1]
            al = cs.collections[ccl].get_paths()[0] # the points constract each line 
            vl = al.vertices
            lonl = vl[:,0]
            depthl = vl[:,1]
            latl = np.repeat(py,len(lonl))
            lowerlon.append(lonl)
            lowerlat.append(latl)
            lowerdepth.append(depthl)
        else:
            lon = data.variables['Longitude'][idx-1:idx2+1]#[-88.52002 , -88.47998 , -88.44    , -88.400024, -88.359985, -88.32001 ,-88.28003 , -88.23999 , -88.20001 , -88.160034, -88.119995, -88.08002 ,-88.03998 , -88.      , -87.96002 , -87.91998 , -87.880005, -87.839966]#data.variables['Longitude'][237:255]
            depth = np.repeat(np.nan,len(lon))
            depth = np.ma.masked_array(depth,mask=np.nan)            
            lat = np.repeat(py,len(lon))

            upperlon.append(lon)
            upperlat.append(lat)
            upperdepth.append(depth)
        # the lower depth 
            lonl = data.variables['Longitude'][idx-1:idx2+1]#[-88.52002 , -88.47998 , -88.44    , -88.400024, -88.359985, -88.32001 ,-88.28003 , -88.23999 , -88.20001 , -88.160034, -88.119995, -88.08002 ,-88.03998 , -88.      , -87.96002 , -87.91998 , -87.880005, -87.839966]#data.variables['Longitude'][237:255]
            depthl = np.repeat(np.nan,len(lonl))
            depthl = np.ma.masked_array(depthl,mask=np.nan)
            latl = np.repeat(py,len(lonl))
            lowerlon.append(lonl)
            lowerlat.append(latl)
            lowerdepth.append(depthl)               
    py =  data.variables['Latitude'][idx3:idx4]
    px = data.variables['Longitude'][idx-1:idx2+1]  
    upperdepth = np.array(upperdepth)
    upperlon = np.array(upperlon)
    upperlat = np.array(upperlat)
    upperdepth = np.hstack(upperdepth)
    upperdepth = np.ma.masked_array(upperdepth,mask=np.nan)
    upperlon =np.hstack( upperlon)
    upperlat = np.hstack(upperlat)
    lowerdepth = np.array(lowerdepth)
    lowerlon = np.array(lowerlon)
    lowerlat = np.array(lowerlat)
    lowerdepth = np.hstack(lowerdepth)
    lowerdepth = np.ma.masked_array(lowerdepth,mask=np.nan)     
    lowerlon =np.hstack( lowerlon)
    lowerlat = np.hstack(lowerlat)
    [px,py]=np.meshgrid(px,py)

    points = zip(upperlon,upperlat)
    points2 = zip(lowerlon,lowerlat)
    print len(points)
    print len(upperdepth)
    Z = griddata(points,upperdepth,(px,py),method='linear')  
    zz = []  
    zz1 = []
    Z1 = griddata(points2,lowerdepth,(px,py),method='linear')
    for i in range(len(Z)):
        Z[i]=  np.ma.masked_array(Z[i],mask=np.nan)          
        zz.append(Z[i])    
    for i in range(len(Z1)):
        Z1[i]=  np.ma.masked_array(Z1[i],mask=np.nan)          
        zz1.append(Z1[i])
    return px,py,Z,Z1

def multicore1(a,parameter):
        pool = mp.Pool(15)
        res = pool.map(partial(IniLikelihood,a),parameter)
        return res

def multicore2(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integ,a),loc)
        return res

def multicore4(a,locs):
        pool = mp.Pool(15)
        res = pool.map(partial(depthcalculate,a),locs)
        return res

def multicore5(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integcontinuous,a),loc)
        return res

def multicore_para(parameter,a):
        pool = mp.Pool(15)
        parameter=np.array(parameter)
        res = pool.map(partial(calculate,a=a),parameter)
        return res

def multicore_parameter(parameter,a):
        pool = mp.Pool(15)
        parameter=np.array(parameter)
        res = pool.map(partial(calculate_continuous,a=a),parameter)
        return res

def multicore6(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integcfconti,a),loc)
        return res

def multicore9(a,loc):
        pool = mp.Pool(15)
        res = pool.map(partial(integcfcontiins,a),loc)
        return res

def multicore7(parameter,a):
        pool = mp.Pool(15)
        res = pool.map(partial(IniLikelihoodcal,a),parameter)
        return res

def multicore8(parameter,a):
        pool = mp.Pool(15)
        res = pool.map(partial(IniLikelihoodcalfield,a),parameter)
        return res

def extract_key(v):
        return v[0]

def upNetCDF(a,timeindex,depthindex):  # when click Submerged hydrodynamic Upload, OSCAR part 
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
    jj = timeindex
    mm = depthindex
    zl = a.px[tuple(a.index[jj][mm])]
    zla = a.py[tuple(a.index[jj][mm])]
    zcon = a.Z[jj,mm][tuple(a.index[jj][mm])]
    priordep.append(a.dep[mm])
    priorT.append(a.OSCARt[jj])
    priorlon.append(zl)   # all depth available lon; (len(Time)*len(depth)
    priorlat.append(zla)
    priorcon.append(zcon)  # first depth=20 is the 1st time; total has 15 times; total number of priorcon is 600; the first 300 belongs to the 1st prediction time 

    clat = []
    clong = []
    ccon = []
    ct = []
    conc = []
    depth = []
    priorx = []
    priory = []

    for p in range(len(priorlon[0])):
        clat.append(priorlat[0][p])
        clong.append(priorlon[0][p])
        ccon.append(priorcon[0][p])
        depth.append(priordep[0])            
        ct.append(a.OSCARt[timeindex])  # change later time 
    if len(clat)>0:
        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(clat,clong)])
        priorx.append(np.array(map(float,camdatalist[:,0]))/1000)
        priory.append(np.array(map(float,camdatalist[:,1]))/1000)
    if len(clat) == 0:  
        priorx = [[0]]
        priorx = np.array(priorx)
        priory = [[0]]
        priory = np.array(priory)        
        clat = [0]
        clat = np.array(clat)
        clong = [0]
        clong = np.array(clong)
        ccon = [0]
        ccon = np.array(ccon)
        ct = [0]
        ct = np.array(ct)
        depth = [0]
        depth = np.array(depth)      
    data = zip(clat,clong,priorx[0],priory[0],ccon,ct,depth)       
    return data

class Preliminars: # SOSim
    def __init__(self): 
        self.valid = 0.0

class soscore(Preliminars):
    def __init__(self,datalist):
        Preliminars.__init__(self)
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

        lat = np.array(lat)[~np.isnan(lat)]
        lon = np.array(lon)[~np.isnan(lon)]
        SpillT = np.array(SpillT)[~pd.isnull(SpillT)]
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

        durationt = [CalTime(SpillT[0],SpillT[1])]
        smallspill = np.linspace(0,durationt[0],durationt[0])
        dura = np.pad(durationt,(0,1),'constant')

        PTi = []
        predictTime = []
        uni = []
        t = [CalTime(SpillT[0],PredictT[vld]) for vld in range(len(PredictT))]
        newt = [list(j) for i, j in groupby(t)]  # group the sample time into the campaign number 
        cc = np.unique(newt)        
        lent = [len(list(j)) for i, j in groupby(t)]        
        d = np.cumsum(lent)

        for i in range(len(newt)):
            cc = np.unique(newt[i])
            uni.append(cc)

        for j in range(len(uni)): 
            if j ==0 :      
                predictTime.append(uni[j])
            else:
                aa = uni[j]-uni[j-1]
                predictTime.append(aa)
        #print 'predictTime',predictTime
        #print t 

        self.x01 = self.x02 = self.x03 = self.x04 = x0
        self.y01 = self.y02 = self.y03 = self.y04 = y0
        self.xx01 = self.xx02 = self.xx03 = self.xx04 = x0
        self.yy01 = self.yy02 = self.yy03 = self.yy04 = y0        
        self.SpillT = SpillT
        self.PredictT = PredictT
        self.Scale = Scale
        self.xNode = Node[0]
        self.yNode = Node[1]
        self.OilType = OilType
        self.sigmax01 = self.sigmax02 = self.sigmax03 = self.sigmax04 = sigmax0
        self.sigmay01 = self.sigmay02 = self.sigmay03 = self.sigmay04 = sigmay0
        self.Dx01 = self.Dx02 = self.Dx03 = self.Dx04 = sigmax0
        self.Dy01 = self.Dy02 = self.Dy03 = self.Dy04 = sigmay0        
        self.DDx01 = self.DDx02 = self.DDx03 = self.DDx04 = sigmax0
        self.DDy01 = self.DDy02 = self.DDy03 = self.DDy04 = sigmay0  
        self.x00 = lat
        self.y00 = lon
        self.t = t # t is prediction time 
        self.scale = Scale
        self.lat0 = lat[0]
        self.lon0 = lon[0]
        self.dura = dura
        self.PT = predictTime
        self.smallspill = smallspill

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
        sst = []
        Modellat = []
        Modellon = []
        uni = []

        for i in range(len(CampaignFileName)):
            if i == 0:
                campdata = pd.read_csv(CampaignFileName[i])
            else:
                campdata = pd.concat([campdata , pd.read_csv(CampaignFileName[i])],axis = 0,ignore_index=True)                

        SampleT = np.array(campdata['SampleTime'])    
        DLlat = np.array(campdata["Lat"])
        DLlon = np.array(campdata["Lon"])
        DLc = np.array(campdata["Con"])
        DLplotcon = DLc*0.01
        depth = np.array(campdata["depth"])
        s.extend(SampleT)

        for j in range(len(s)):  
            cc = CalTime(self.SpillT[0],s[j])
            st.append(cc)  # st is all sampling time        

        stuni = np.unique(st)     
        DLxx = []
        DLyy = []
        DLccon = []
        ssti = []
        YY = []
        cc = []
        ee = []
        N = len(DLlat)    
        for s in DLc:
            if s == 0.0:
                conValue = (0.001)
            else:
                conValue = (s)
            DLcon.append(conValue)

        coo = zip(DLlat,DLlon)
        coocon = zip(coo,DLcon,st)
        extract_key = lambda x:x[0]
        datap = sorted(coocon,key=extract_key)       
        resultp = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datap, extract_key)] # data group together 
        DLclp=[]
        fp=[]
        ttp=[]
        da = []
        fileddep = []
        allfield = []
        for i in range(len(resultp)):
            conres=[list(p) for m, p in itertools.groupby(resultp[i][1],lambda x:x[1])]
            for j in conres:
                jarray=np.array(j)
                con = np.mean(jarray[:,0])
                DLclp.append(con)            
                h = resultp[i][0]
                fp.append(h)
                ttp.append(j[0][1])
        fp = np.array(fp)   # location 
        DLclp = np.array(DLclp) #concentration 
        ttp = np.array(ttp)     
        totalt,totalx,totaly,totalcon = (list(t) for t in zip(*sorted(zip(ttp,fp[:,0],fp[:,1],DLclp))))
        camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(totalx,totaly)])
        DLx.append(np.array(map(float,camdatalist[:,0]))/1000)  # location x
        DLy.append(np.array(map(float,camdatalist[:,1]))/1000)  # location y    
        DLx = DLx[0]
        print 'sample_camp_qfielddataDLx',DLx
        print 'sample_camp_st',totalt
        DLy = DLy[0]
        DLclp = DLclp
        ttp =ttp
        self.N = N
        self.stuni =stuni
        self.fieldDLx = DLx  # 363
        self.fieldDLy = DLy # 3179
        self.fieldDLcon = DLclp
        self.st = totalt
        self.DLlatfield = DLlat
        self.DLlonfield = DLlon
        self.DLconfield = DLplotcon
        self.DLlat = fp[:,0]
        self.DLlon = fp[:,1]

    def priorandfield(self):  # upload Campaign part
        timegap = len(range(int(np.max(self.st) - np.min(self.st))))+1 
        time = np.linspace(np.min(self.st),np.max(self.st),timegap)

        time = time[time>=np.min(self.ttp)]
        time = time[time<=np.max(self.ttp)]
        print 'timeme',time 
        priorlat = []
        priorlon = []
        priorcon = []
        priorx = []
        priory = []
        priort = []
        sampletime = []   
        DLxx = []
        DLyy = []
        DLccon = []
        ssti = []             
        if len(time)>0: 
            for i in range(len(time)):
                self.priorsampletime = time[i]
                priorlat1,priorlon1,priorcon1,priorx1,priory1,priort1 = sampleprior(self)
                priorlat = np.append(priorlat,priorlat1)
                priorlon = np.append(priorlon,priorlon1)
                priorcon = np.append(priorcon,priorcon1)
                priorx = np.append(priorx,priorx1)
                priory = np.append(priory,priory1)
                priort = np.append(priort,priort1)  
            print 'priorlat',priorlat
            priortt,priorlati,priorlong,priorconc = (list(t) for t in zip(*sorted(zip(priort,priorlat,priorlon,priorcon))))            
            priortt = np.array(priortt)
            priorlati = np.array(priorlati)
            priorlong = np.array(priorlong)
            priorconc = np.array(priorconc)  
        else: 
            priorx = []
            priory = []
            priorcon = []
            priorlat = []
            priorlon = []
            priort = []
            priorconc = []
            priorlati = []
            priorlong = []
            priortt = []
        DLx=np.append(self.fieldDLx,priorx)
        DLy=np.append(self.fieldDLy,priory)
        DLcon=np.append(self.fieldDLcon,priorcon)
        DLlat = np.append(self.DLlat,priorlat)
        DLlon = np.append(self.DLlon,priorlon)
        st=np.append(self.st,priort)
        plotpriorcon = np.array(priorconc)*0.01

        totalt,totalx,totaly,totalcon = (list(t) for t in zip(*sorted(zip(st,DLx,DLy,DLcon))))
        uni = np.unique(totalt)

        newst = [list(j) for i, j in groupby(totalt)]  # group the sample time into the campaign number 
        lenst = [len(list(j)) for i, j in groupby(totalt)]      
        d = np.cumsum(lenst)

        for j in range(len(uni)): 
            if j ==0 :      
                sampletime.append(uni[j])
            else:
                aa = uni[j]-uni[j-1]
                sampletime.append(aa)

        tt = np.array(st)

        for m in range(len(newst)):
            if m == 0:
                DLxx.append(totalx[m:d[m]])
                DLyy.append(totaly[m:d[m]])
                DLccon.append(totalcon[m:d[m]])
            else:
                DLxx.append(totalx[d[m-1]:d[m]])
                DLyy.append(totaly[d[m-1]:d[m]])
                DLccon.append(totalcon[d[m-1]:d[m]])  

        print 'DLxx',DLxx
        print 'DLyy',DLyy
        self.DLxx = DLxx
        self.DLyy = DLyy
        self.DLccon = DLccon
        self.sst = newst
        self.DLlatprior = priorlati        
        self.DLlonprior = priorlong
        self.ST=sampletime # difference between two sample time
        self.uniST = uni
        self.samplet = st  # raw data sample time 
        self.DLconprior = plotpriorcon
        self.priort = priortt
        self.DLlon = DLlon
        self.DLlat = DLlat
        self.DLcon = DLcon

    def field(self):  # upload Campaign part
        DLx = self.fieldDLx
        DLy = self.fieldDLy
        DLcon = self.fieldDLcon
        DLlat = self.DLlat
        DLlon = self.DLlon
        st = self.st
        plotpriorcon = DLcon*0.01

        totalt,totalx,totaly,totallat,totallon,totalcon = (list(t) for t in zip(*sorted(zip(st,DLx,DLy,DLlat,DLlon,DLcon))))
        uni = np.unique(totalt)

        newst = [list(j) for i, j in groupby(totalt)]  # group the sample time into the campaign number 
        lenst = [len(list(j)) for i, j in groupby(totalt)]      
        d = np.cumsum(lenst)
        sampletime = []
        for j in range(len(uni)): 
            if j ==0 :      
                sampletime.append(uni[j])
            else:
                aa = uni[j]-uni[j-1]
                sampletime.append(aa)

        DLxx = []
        DLyy = []
        DLccon = []
        lat = []
        lon = []
        time = []

        for m in range(len(newst)):
            if m == 0:
                DLxx.append(DLx[m:d[m]])
                DLyy.append(DLy[m:d[m]])
                DLccon.append(DLcon[m:d[m]])
                lat.append(DLlat[m:d[m]])
                lon.append(DLlon[m:d[m]])
                time.append(st[m:d[m]])
            else:
                DLxx.append(DLx[d[m-1]:d[m]])
                DLyy.append(DLy[d[m-1]:d[m]])
                DLccon.append(DLcon[d[m-1]:d[m]])
                lat.append(DLlat[d[m-1]:d[m]])
                lon.append(DLlon[d[m-1]:d[m]])    
                time.append(st[d[m-1]:d[m]])

        alon = np.concatenate(lon,axis=None)
        alat = np.concatenate(lat,axis=None)
        acon = np.concatenate(DLccon,axis=None)
        atime = np.concatenate(time,axis=None)     

        self.DLxx = DLxx
        self.DLyy = DLyy
        self.DLccon = DLccon
        self.sst = newst
        self.ST=sampletime # difference between two sample time
        self.uniST = uni
        self.samplet = st  # raw data sample time 
        self.DLconplot = plotpriorcon
        self.DLlon = DLlon
        self.DLlat = DLlat
        self.DLcon = DLcon

    def writeNetCDF(self):  # record oscar data to a excel file 
        totalresult = []
        eachresult = []
        depthresult = []
        rawdata = []
        # save data to a excel 
        row = 1
        col = 0    
        workbook = xlsxwriter.Workbook('hello.xlsx')
        worksheet = workbook.add_worksheet()    
        worksheet.write(0,0,'latitude')
        worksheet.write(0,1,'longitude')        
        worksheet.write(0,2,'y0')
        worksheet.write(0,3,'x0')
        worksheet.write(0,4,'total_concentration')
        worksheet.write(0,5,'time_after_spill')
        worksheet.write(0,6,'depth')

        timein = np.where(self.OSCARt<=np.max(self.t))
        print 'self.ttt',self.OSCARt
        print 'timein',timein
        for l in range(len(timein[0])):
            for o in range(len(self.dep)):
                result = upNetCDF(self,l,o) 
                for s,k,i,j,p,m,d in result:
                    if abs(s)>0 and abs(k)>0 and abs(i)>0 and abs(j)>0 and abs(p)>0 and abs(m)>0 and abs(d)>0:
                        worksheet.write(row,col,s)
                        worksheet.write(row,col+1,k)                    
                        worksheet.write(row,col+2,i)
                        worksheet.write(row,col+3,j)
                        worksheet.write(row,col+4,p)
                        worksheet.write(row,col+5,m)        
                        worksheet.write(row,col+6,d)                
                        row +=1
        workbook.close()
        # read that excel 
        campdata=pd.read_excel('hello.xlsx',names=None)
        priorlat = np.array(campdata['latitude'])
        priorlon = np.array(campdata["longitude"])
        priorx = np.array(campdata["x0"])
        priory = np.array(campdata["y0"])
        priorcon = np.array(campdata["total_concentration"])
        priortime = np.array(campdata["time_after_spill"])
        depth = np.array(campdata["depth"])  

        coo = zip(priorlat,priorlon)
        coocon = zip(coo,priorcon,priortime)
        extract_key = lambda x:x[0]
        datap = sorted(coocon,key=extract_key)       
        resultp = [[k,[x[1:3] for x in g]] for k, g in itertools.groupby(datap, extract_key)] # data group together 
        DLclp=[]
        fp=[]
        ttp=[]
        da = []
        priorxx = []
        prioryy = []
        allprior = []
        for i in range(len(resultp)):
            conres=[list(p) for m, p in itertools.groupby(resultp[i][1],lambda x:x[1])]
            for j in conres:
                jarray=np.array(j)
                con = np.mean(jarray[:,0])
                DLclp.append(con)            
                h = resultp[i][0]
                fp.append(h)
                ttp.append(j[0][1])
        fp = np.array(fp)   # location 
        DLclp = np.array(DLclp) #concentration 
        ttp = np.array(ttp)
        self.DLclp = DLclp
        self.priorlat = priorlat
        self.priorlon = priorlon
        self.fp = fp
        self.ttp = ttp

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
        print 'mt',mt

        self.priorlon = Mlon
        self.priorlat = Mlat
        self.DLclp = Mcon
        self.ttp = mt   # Tti is prior time divided into different parts 

    def upNet(self,File):  # when click Submerged hydrodynamic Upload, OSCAR part 
        data = Dataset(File,mode='r')
        pt = data.variables['time'][:]
        px = data.variables['longitude'][:]  # 1 dimension 
        py = data.variables['latitude'][:]
        [px,py] = np.meshgrid(px,py)        
        dep=data.variables['depth'][:]
        Z = data.variables['total_concentration'][:]  # 4 dimension: time depth latitude longitude        
        tt = np.zeros(len(pt))
        print 'ttt',tt
        for k in range(len(pt)):   
                tt[k] = (pt[k]/86400.) + 30.        # 30 days is the OSCAR data begin after 30 days               
        print 'ttttt',tt
        index=[]
        for i in range(Z.shape[0]):  # time
            ind1=[]
            for j in range(Z.shape[1]): #depth 
                PP = Z[i,j,:,:]
                PP=ma.masked_values(PP,-99)
                ind = np.where(PP>0)
                ind1.append(ind)
            index.append(ind1)

        self.OSCARt = tt   
        self.dep = dep
        self.Z = Z
        self.pt = pt
        self.px = px
        self.py = py
        self.index = index

    def upNetMulti(self,Filename):
        tt=[] 
        dep=[]
        Z=[]
        pt=[]
        px=[]
        py=[]
        index=[]
        for i in Filename:
            print 'i',i 
            self.upNet(i)
            tt.extend(self.OSCARt)
            # dep.extend(self.dep)
            dep = self.dep
            Z.extend(self.Z)
            pt.extend(self.pt)
            px.extend(self.px)
            py.extend(self.py)
            index.extend(self.index)
        self.OSCARt =np.array(tt)
        self.dep = np.array(dep)
        self.Z = np.array(Z)
        self.pt = np.array(pt)
        self.px = np.array(px)
        self.py = np.array(py)
        self.index = np.array(index)

    def GupNetCDF(self,File):  # when click Submerged hydrodynamic Upload, OSCAR part 
        # read NetCDF 
        print 'File',File
        data = Dataset(File,mode='r')
        px = data.variables['longitude']
        py = data.variables['latitude']
        particle = data.variables['particle_count'][:]
        Z = data.variables['surface_concentration'][:]        
        pt = data.variables['time'][:]
        dep=data.variables['depth'][:]

        priorx = []
        priory = []        

        px=np.array(px)
        py=np.array(py)
        Z = np.array(Z)

        tt = np.zeros(len(pt))
        for k in range(len(pt)):
            tt[k] = (pt[k]/86400.)

        self.priorlon = px
        self.priorlat = py
        self.ttp = tt
        self.DLclp = Z

def submerged_main(myinformation,progressBar):#
# information 
    global ourinformation
    ourinformation = myinformation
    if 'HydroButton' in ourinformation and ourinformation['HydroButton'] != '':
        if  ourinformation['SubmergedType'] == 'OSCAR': 
            filename = [ourinformation['HydroButton']]
        else: 
            filename = ourinformation['HydroButton']

    else:
        filename = "null"
    print 'filename',filename    
    a = soscore(ourinformation)
    a.ourinformation = ourinformation    
    a.myinformation = ourinformation
    progressBar.setValue(15)

    if ourinformation['SubmergedType'] == 'OSCAR':
        a.UploadCampaign(myinformation['CampaignButton'])        
        a.upNetMulti(filename)        
        a.writeNetCDF()
        a.priorandfield()

    if ourinformation['SubmergedType'] == 'Nodate':
        a.UploadCampaign(myinformation['CampaignButton'])        
        a.field()

    elif ourinformation['SubmergedType'] == 'Other':
        a.UploadCampaign(myinformation['CampaignButton'])  
        a.upprior(filename)
        a.priorandfield()

    elif ourinformation['SubmergedType'] == 'GNOME':
        a.UploadCampaign(myinformation['CampaignButton'])          
        a.GupNetCDF(filename)
        a.priorandfield()
# data 
    lat0 = a.lat0  # transform parameters 
    lon0 = a.lon0
# prediction area 
    Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.xNode+1)
    Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.yNode+1)
    [XXa,YYa] = np.meshgrid(Xa,Ya)
    xna = np.concatenate(XXa)
    yna = np.concatenate(YYa)

    Xxa = np.linspace(a.lat0+a.scale[0],a.lat0-a.scale[0],a.xNode+1)
    Yya = np.linspace(a.lon0+a.scale[1],a.lon0-a.scale[1],a.yNode+1)
    [XXa,YYa] = np.meshgrid(Xxa,Yya)
    xxa = np.concatenate(XXa)
    yya = np.concatenate(YYa)

    coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(Xa,Ya)])
    xa = np.array(map(float,coord[:,0]))/1000
    ya = np.array(map(float,coord[:,1]))/1000
    [Xx,Yy] = np.meshgrid(xa,ya)
    x = np.concatenate(Xx)
    y = np.concatenate(Yy)
  
    a.x=x  # prediction area
    a.y=y # rediction 
    N = 1000000
    paraindi = []
    MaxLog = []

    a.xx01=[]
    a.yy01=[]
    a.xx02=[]
    a.yy02=[]
    a.xx03=[]
    a.yy03=[]
    a.xx04=[]
    a.yy04=[]    
    a.ssigmax01 = []
    a.ssigmax02 = []
    a.ssigmax03 = []
    a.ssigmax04 = []
    a.ssigmay01 = []
    a.ssigmay02 = []
    a.ssigmay03 = []
    a.ssigmay04 = []
    combinepara=[]
    combinepos = []
    postcombinepara = []
    STT=[]
    x01,y01,x02,y02,x03,y03,x04,y04,sigmax01,sigmay01,sigmax02,sigmay02,sigmax03,sigmay03,sigmax04,sigmay04=\
        a.x01,a.y01,a.x02,a.y02,a.x03,a.y03,a.x04,a.y04,a.sigmax01,a.sigmay01,a.sigmax02,a.sigmay02,a.sigmax03,a.sigmay03,a.sigmax04,a.sigmay04 
    
    if ourinformation['Run'] == 'Run':    
        if ourinformation['Type'] == 'continuous':
            a.stind = np.transpose(a.uniST)  # sample time  
            a.stind = np.insert(a.stind,0,-1)   
            print 'a.stind',a.stind           
            DDatax = []
            DDatay = []
            DDatacon = []
            td = []
            priorxx = []
            prioryy = []
            priorccon = []
            priorYY = []
            for k in range(len(a.DLxx)):
                a.DLx = a.DLxx[k]
                a.DLy = a.DLyy[k]                  
                a.DLcon = a.DLccon[k] 
                DDLx = a.DLx
                DDLy = a.DLy
                DDLcon = a.DLcon
                a.st = a.sst[k][0] 
             
                a.fieldDLx = np.around(a.fieldDLx,decimals=8)
                a.fieldDLy = np.around(a.fieldDLy,decimals=8)
                a.fieldDLcon = np.around(a.fieldDLcon,decimals=7) 

                Datax = []
                Datay = []
                Datacon = []

                if ourinformation['SubmergedType']== "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':
                    for l in range(len(a.fieldDLx)):                
                        s = a.fieldDLx[l]
                        ss = a.fieldDLy[l]
                        sss = a.fieldDLcon[l]                           
                        if (s in (np.around(a.DLx,decimals=8))) and (ss in np.around(a.DLy,decimals=8)) and (sss in np.around(a.DLcon,decimals=7)):
                            Datax.append(s)
                            Datay.append(ss)
                            Datacon.append(sss)
                            DDLx = np.delete(DDLx,np.where(np.around(DDLx,decimals=8) ==s))
                            DDLy = np.delete(DDLy,np.where(np.around(DDLy,decimals=8) ==ss))
                            DDLcon = np.delete(DDLcon,np.where(np.around(DDLcon,decimals=7) ==sss))

                    if len(Datax)>0:
                        d=int(len(DDLx)/len(Datax))
                    else:
                        d = 1
                    a.l = d
                    td.append(d) 
                if ourinformation['SubmergedType']== "Nodate":  
                    for l in range(len(a.fieldDLx)):                
                        s = a.fieldDLx[l]
                        ss = a.fieldDLy[l]
                        sss = a.fieldDLcon[l]                           
                        if (s in (np.around(a.DLx,decimals=8))) and (ss in np.around(a.DLy,decimals=8)) and (sss in np.around(a.DLcon,decimals=7)):
                            Datax.append(s)
                            Datay.append(ss)
                            Datacon.append(sss)                    

                    a.Datax = Datax # field data 
                    a.Datay = Datay
                    a.Datacon = Datacon

                STI = a.smallspill[a.smallspill>=a.stind[k]]
                STI = STI[STI<a.stind[k+1]]  # the divided small spill within the sample time 
                print 'STI',STI
                t=a.stind[k+1]           
                STT=np.append(STT,STI)  # the old and new time 
                a.STT = STT       
 
                a.xx01=np.append(a.xx01,[a.x01]*len(STI))
                a.xx02=np.append(a.xx02,[a.x02]*len(STI))   
                a.xx03=np.append(a.xx03,[a.x03]*len(STI))
                a.xx04=np.append(a.xx04,[a.x04]*len(STI))                     
                a.yy01=np.append(a.yy01,[a.y01]*len(STI))
                a.yy02=np.append(a.yy02,[a.y02]*len(STI))   
                a.yy03=np.append(a.yy03,[a.y03]*len(STI))
                a.yy04=np.append(a.yy04,[a.y04]*len(STI))                    
                a.ssigmax01=np.append(a.ssigmax01,[a.sigmax01]*len(STI))
                a.ssigmax02=np.append(a.ssigmax02,[a.sigmax02]*len(STI))   
                a.ssigmax03=np.append(a.ssigmax03,[a.sigmax03]*len(STI))
                a.ssigmax04=np.append(a.ssigmax04,[a.sigmax04]*len(STI))                     
                a.ssigmay01=np.append(a.ssigmay01,[a.sigmay01]*len(STI))
                a.ssigmay02=np.append(a.ssigmay02,[a.sigmay02]*len(STI))   
                a.ssigmay03=np.append(a.ssigmay03,[a.sigmay03]*len(STI))
                a.ssigmay04=np.append(a.ssigmay04,[a.sigmay04]*len(STI))                          
                posss = a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04,STT
                combinepos.append(posss) 

                parameter = sampler(N)
                if ourinformation['SubmergedType']== "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':   
                    a.priorxx = DDLx
                    priorxx.append(DDLx)                    
                    a.prioryy = DDLy
                    prioryy.append(DDLy)                 
                    a.priorccon = DDLcon
                    priorccon.append(DDLcon)
                    a.Datax = Datax
                    DDatax.append(Datax)
                    a.Datay = Datay
                    DDatay.append(Datay)
                    a.Datacon = Datacon
                    DDatacon.append(Datacon)
                                     
                    if len(a.Datax) == 0:             
                        prob,a.MaxLogLike = priorlikelihood(a,N)
                    if len(a.Datax) > 0 and len(a.priorxx) >0:                    
                        prob,a.MaxLogLike = fieldpriorlikelihood(a,N)
                    if len(a.Datax) > 0 and len(a.priorxx) == 0 : 
                        prob,a.MaxLogLike = fieldlikelihood(a,N)                    
                    a.r = parameter[np.argmax(prob)]

                if ourinformation['SubmergedType']== "Nodate":  
                    prob,a.MaxLogLike = fieldlikelihood(a,N)                    
                    print 'loc2___',np.argmax(prob)            
                    a.r = parameter[np.argmax(prob)]

                MaxLog.append(a.MaxLogLike) 
                combinepara.append(a.r)  

                (vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4) = combinepara[k]                         
                a.xx01 = a.xx01 + vx1*(t-STT)
                a.yy01 = a.yy01 + vy1*(t-STT)   
                a.xx02 = a.xx02 + vx2*(t-STT)
                a.yy02 = a.yy02 + vy2*(t-STT)    
                a.xx03 = a.xx03 + vx3*(t-STT)
                a.yy03 = a.yy03 + vy3*(t-STT)   
                a.xx04 = a.xx04 + vx4*(t-STT)
                a.yy04 = a.yy04 + vy4*(t-STT)  

                a.ssigmax01 = a.ssigmax01 + np.sqrt(2*Dx1*(t-STT))
                a.ssigmay01 = a.ssigmay01 + np.sqrt(2*Dy1*(t-STT))                  
                a.ssigmax02 = a.ssigmax02 + np.sqrt(2*Dx2*(t-STT))
                a.ssigmay02 = a.ssigmay02 + np.sqrt(2*Dy2*(t-STT)) 
                a.ssigmax03 = a.ssigmax03 + np.sqrt(2*Dx3*(t-STT))
                a.ssigmay03 = a.ssigmay03 + np.sqrt(2*Dy3*(t-STT))                  
                a.ssigmax04 = a.ssigmax04 + np.sqrt(2*Dx4*(t-STT))
                a.ssigmay04 = a.ssigmay04 + np.sqrt(2*Dy4*(t-STT)) 

                pos = a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04
                a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04 = pos
                STT = np.ones(len(STT))*a.stind[k+1]#SampleTime[i+1]   
                poos = a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04,STT          
                postcombinepara.append(poos)
                print 'combinepara',combinepara
            np.savetxt('combinepara.txt',combinepara,delimiter=',')
            np.savetxt('MaxLogLike.txt',MaxLog,delimiter=',')  
            np.savetxt('postcombinepara.txt',postcombinepara[-1],delimiter=',')  
            np.save('combinepos.npy',combinepos)  
        
        if ourinformation['Type'] == 'instantaneous':
            td = []
            for k in range(len(a.DLxx)): # a.DLxx is the field data 
                a.DLx = a.DLxx[k]  # field and prior
                print 'a.DLx',a.DLx
                a.DLy = a.DLyy[k]
                a.DLcon = a.DLccon[k]
                a.st = a.sst[k][0]
                print 'a.st',a.st
                a.STT = [0]
                t = a.ST[k]
                cc = []
                index_sets = [np.argwhere(i==a.DLx) for i in np.unique(a.DLx)]
                DDLx = a.DLx
                DDLy = a.DLy
                DDLcon = a.DLcon
                a.fieldDLx = np.around(a.fieldDLx,decimals=8) # field data in km
                a.fieldDLy = np.around(a.fieldDLy,decimals=8)
                a.fieldDLcon = np.around(a.fieldDLcon,decimals=7)    
                Datax = []
                Datay = []
                Datacon = []
                DataY = []                 
                if ourinformation['SubmergedType']== "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':
                    for l in range(len(a.fieldDLx)):                 
                        s = a.fieldDLx[l]  # field only 
                        ss = a.fieldDLy[l]
                        sss = a.fieldDLcon[l] 
                        if (s in (np.around(a.DLx,decimals=8))) and (ss in np.around(a.DLy,decimals=8)) and (sss in np.around(a.DLcon,decimals=7)):
                            Datax.append(s)  # Datax is field data 
                            print 'datax',Datax
                            Datay.append(ss)
                            Datacon.append(sss)
                            DDLx = np.delete(DDLx,np.where(np.around(DDLx,decimals=8) ==s))
                            DDLy = np.delete(DDLy,np.where(np.around(DDLy,decimals=8) ==ss))
                            DDLcon = np.delete(DDLcon,np.where(np.around(DDLcon,decimals=7) ==sss))

                    print 'Datax',Datax 
                    print 'DDLx',DDLx # 
                    if len(DDLx)>0 and len(Datax)>0:
                        d=int(len(DDLx)/len(Datax))
                    else: 
                        d = 1 
                    a.l = d 
                    td.append(d)                       
                if k ==0:           
                    a.xx01 = [a.x01]
                    a.yy01 = [a.y01]
                    a.xx02 = [a.x02]
                    a.yy02 = [a.y02]
                    a.xx03 = [a.x03]
                    a.yy03 = [a.y03]
                    a.xx04 = [a.x04]
                    a.yy04 = [a.y04]                                
                    a.ssigmax01=[a.sigmax01]
                    a.ssigmax02=[a.sigmax02]  
                    a.ssigmax03=[a.sigmax03]
                    a.ssigmax04=[a.sigmax04]                  
                    a.ssigmay01=[a.sigmay01]
                    a.ssigmay02=[a.sigmay02]
                    a.ssigmay03=[a.sigmay03]
                    a.ssigmay04=[a.sigmay04]
                    posss = a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04
                    combinepos.append(posss) 
                if ourinformation['SubmergedType']== "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':     
                    a.priorxx = DDLx
                    a.prioryy = DDLy
                    a.priorccon = DDLcon
                    a.Datax = Datax
                    a.Datay = Datay
                    a.Datacon = Datacon
                    parameter = sampler(N)                      
                    if len(a.Datax) == 0:             
                        prob,a.MaxLogLike = priorlikelihood(a,N)
                    if len(a.Datax) > 0 and len(a.priorxx) >0:                    
                        prob,a.MaxLogLike = fieldpriorlikelihood(a,N)
                    if len(a.Datax) > 0 and len(a.priorxx) == 0 : 
                        prob,a.MaxLogLike = fieldlikelihood(a,N)   
                    a.r = parameter[np.argmax(prob)] 

                if ourinformation['SubmergedType']== "Nodate":      
                    prob,a.MaxLogLike = Likelihood(a,N)  
                    print 'MaxLogLike',a.MaxLogLike                      
                    parameter = sampler(N)  
                    MaxLog.append(a.MaxLogLike)
                    a.r = parameter[np.argmax(prob)]      

                combinepara.append(a.r)  
                MaxLog.append(a.MaxLogLike)      
                print 'a.r',a.r
                vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.r    
                print 'vx1',vx1
                #a.MaxLogLike = (-1)*min(rstmp)   
                a.xx01 = [a.xx01[0] + vx1*t]
                a.yy01 = [a.yy01[0] + vy1*t]
                a.xx02 = [a.xx02[0] + vx2*t]
                a.yy02 = [a.yy02[0] + vy2*t]
                a.xx03 = [a.xx03[0] + vx3*t]
                a.yy03 = [a.yy03[0] + vy3*t]
                a.xx04 = [a.xx04[0] + vx4*t]
                a.yy04 = [a.yy04[0] + vy4*t]    
                a.ssigmax01 = [a.ssigmax01[0] + np.sqrt(2*Dx1*t)]
                a.ssigmay01 = [a.ssigmay01[0] + np.sqrt(2*Dy1*t)]                 
                a.ssigmax02 = [a.ssigmax02[0] + np.sqrt(2*Dx2*t)]
                a.ssigmay02 = [a.ssigmay02[0] + np.sqrt(2*Dy2*t)]
                a.ssigmax03 = [a.ssigmax03[0] + np.sqrt(2*Dx3*t)]
                a.ssigmay03 = [a.ssigmay03[0] + np.sqrt(2*Dy3*t)]                 
                a.ssigmax04 = [a.ssigmax04[0] + np.sqrt(2*Dx4*t)]
                a.ssigmay04 = [a.ssigmay04[0] + np.sqrt(2*Dy4*t)]   
                poos = a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04         
                postcombinepara.append(poos)
                print 'postcombinepara',postcombinepara
        np.savetxt('combinepara.txt',combinepara,delimiter=',')
        np.savetxt('MaxLogLike.txt',MaxLog,delimiter=',')  
        np.savetxt('postcombinepara.txt',postcombinepara[-1],delimiter=',')  

    if ourinformation['Run'] == 'Recalc':      
        if ourinformation['Type'] == 'continuous':     
	        MaxLog = np.loadtxt('MaxLogLike.txt',delimiter=',')
	        combinepara = np.loadtxt('combinepara.txt',delimiter=',')   
	        postcombinepara = np.loadtxt('postcombinepara.txt',delimiter=',')
	        combinepos = np.load('combinepos.npy')
        if ourinformation['Type'] == 'instantaneous':  
            combinepara = np.loadtxt('combinepara.txt',delimiter=',')
            MaxLog = np.loadtxt('MaxLogLike.txt',delimiter=',')
            postcombinepara = np.loadtxt('postcombinepara.txt',delimiter=',')
        print 'postcombinnnne',postcombinepara          
        print 'combinepara',combinepara
    progressBar.setValue(50)
    if ourinformation['contour'] == 'contour':
        if ourinformation['spillname'] == 'DWH':
            ppltpx = []
            ppltpy = []
            ppltZ = []
            ppltZ1 = []
            for i in range(len(a.t)):
                if a.t[i]<=41 and a.t[i]>=0: 
                    file = ['2010_900.nc','2010_1000.nc','2010_1100.nc','2010_1200.nc','2010_1300.nc','2010_1400.nc','2010_1500.nc']
                    timedf = a.t[i]
                    a.tt = int(timedf)# May26
                    print 'a.tt',a.tt
                    pltpx,pltpy,pltZ,pltZ1 = plot(a.tt,file,Xa,Ya) # 
                    ppltpx.append(pltpx)
                    ppltpy.append(pltpy)
                    ppltZ.append(pltZ)
                    ppltZ1.append(pltZ1)                      
                if a.t[i]>41 and a.t[i]<=71: 
                    file = ['900.nc','1000.nc','1100.nc','1200.nc','1300.nc']
                    timedf = a.t[i]-45
                    a.tt = int(5 + timedf) # June     
                    pltpx,pltpy,pltZ,pltZ1 = plot(a.tt,file,Xa,Ya) # 
                    ppltpx.append(pltpx)
                    ppltpy.append(pltpy)
                    ppltZ.append(pltZ)
                    ppltZ1.append(pltZ1)          
        else:
            Xa = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.xNode+1)
            Ya = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.yNode+1)     
            [xd,yd] = np.meshgrid(Xa,Ya)
            xd = np.concatenate(xd)
            yd = np.concatenate(yd)
            lcs = zip(xd,yd)                    
            z = np.array(multicore4(a,lcs))
            zmax = z[:,0]
            zmin = z[:,1]
            zminf = zmin
            zmaxf = zmax            
            xaa = Xa
            yaa = Ya                    

#-------------------------pending----     
    if ourinformation['Type'] == 'instantaneous':
        res = []        
        pret = []
        for u in range(len(a.t)):
            a.tt = a.t[u] # predict time 
            print 'a.tt',a.tt
            diff = []
            diff = a.tt - a.uniST # predict - sampletime 
            print 'diff',diff
            index = np.where(diff>=0)
            a.r = combinepara[np.min(index):np.max(index)+1]
            timed=diff[diff>=0]
            ttt=a.tt-timed
            ttt=np.insert(np.diff(ttt),0,ttt[0])
            print 'ttt',ttt
            if diff[-1]>0:
                ttt=np.append(ttt,a.tt-a.uniST[-1])
                pret.append(ttt)
                t = ttt
                print 'ttt2',t
            for i in range(len(t)):
                if i >= len(a.r):
                    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = combinepara[-1]
                else:                 
                    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = combinepara[np.max(index)]
                print 'a.r',a.r
                if i == 0:
                    a.xx01 = a.x01 + vx1*t[i]
                    a.yy01 = a.y01 + vy1*t[i]
                    a.xx02 = a.x02 + vx2*t[i]
                    a.yy02 = a.y02 + vy2*t[i]
                    a.xx03 = a.x03 + vx3*t[i]
                    a.yy03 = a.y03 + vy3*t[i]
                    a.xx04 = a.x04 + vx4*t[i]
                    a.yy04 = a.y04 + vy4*t[i]     
                    print 'a.xx01',a.xx01
                else:
                	print 'a.xxo1',a.xx01
                	print 'vx1',vx1
                	print 'ti',t[i]
                	a.xx01 = a.xx01 + vx1*t[i]
                	a.yy01 = a.yy01 + vy1*t[i]
                	a.xx02 = a.xx02 + vx2*t[i]
                	a.yy02 = a.yy02 + vy2*t[i]
                	a.xx03 = a.xx03 + vx3*t[i]
                	a.yy03 = a.yy03 + vy3*t[i]
                	a.xx04 = a.xx04 + vx4*t[i]
                	a.yy04 = a.yy04 + vy4*t[i] 
                a.Dx01 = a.Dx01 + np.sqrt(2*Dx1*t[i])
                a.Dy01 = a.Dy01 + np.sqrt(2*Dy1*t[i])                  
                a.Dx02 = a.Dx02 + np.sqrt(2*Dx2*t[i])
                a.Dy02 = a.Dy02 + np.sqrt(2*Dy2*t[i]) 
                a.Dx03 = a.Dx03 + np.sqrt(2*Dx3*t[i])
                a.Dy03 = a.Dy03 + np.sqrt(2*Dy3*t[i])                  
                a.Dx04 = a.Dx04 + np.sqrt(2*Dx4*t[i])
                a.Dy04 = a.Dy04 + np.sqrt(2*Dy4*t[i])  
            a.ttt = a.r[-1]  
            loc = zip(x,y)     
            resa = np.array(multicore2(a,loc))
            resa = np.transpose(resa)
            res.append(resa)
        res = np.array(res)
        s = res/np.max(res) 
    progressBar.setValue(75)
#________________confidence bounds____________
    if ourinformation['Method'] == 'Minimum':  
        bounds1=[(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vymin']),float(ourinformation['vymax']))]   # velocity bounds  vx1, vy1, vx2, vy2
        bounds2=[(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dymin']),float(ourinformation['dymax']))]    # diffusion bounds Dx1,Dy1, Dx2, Dy2
        bounds3=[(-0.999,0.999),(-0.999,0.999),(-0.999,0.999),(-0.999,0.999)] # roh 
        bounds11=[(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vymin']),float(ourinformation['vymax'])),(float(ourinformation['vxmin']),float(ourinformation['vxmax'])),(float(ourinformation['vymin']),float(ourinformation['vymax']))]   # velocity bounds  vx3, vy3, vx4, vy4      
        bounds33=[(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dymin']),float(ourinformation['dymax'])),(float(ourinformation['dxmin']),float(ourinformation['dxmax'])),(float(ourinformation['dymin']),float(ourinformation['dymax']))]  # dy      # diffusion bounds

        a.par = []   
        rescf = []      
        if ourinformation['Run'] == 'Run':   
            if ourinformation['Type'] == 'instantaneous':
                for k in range(len(a.DLxx)):
                    a.DLx = a.DLxx[k]
                    a.DLy = a.DLyy[k]             
                    a.DLcon = a.DLccon[k]              
                    a.st = a.sst[k][0]              
                    a.TTi = [0]
                    t = a.ST[k]
                    a.rr = combinepara[k]
                    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = combinepara[k]
                    a.MaxLogLike = MaxLog[k]
                    if k == 0:
                        a.xx01 = [a.x01]
                        a.yy01 = [a.y01]
                        a.xx02 = [a.x02]
                        a.yy02 = [a.y02]
                        a.xx03 = [a.x03]
                        a.yy03 = [a.y03]
                        a.xx04 = [a.x04]
                        a.yy04 = [a.y04]     
                        a.ssigmax01 = [a.DDx01]
                        a.ssigmay01 = [a.DDy01]                  
                        a.ssigmax02 = [a.DDx02]
                        a.ssigmay02 = [a.DDy02]
                        a.ssigmax03 = [a.DDx03]
                        a.ssigmay03 = [a.DDy03]                 
                        a.ssigmax04 = [a.DDx04]
                        a.ssigmay04 = [a.DDy04]  
                    else:
                        a.xx01 = [a.xx01[0] + vx1*t]
                        a.yy01 = [a.yy01[0] + vy1*t]
                        a.xx02 = [a.xx02[0] + vx2*t]
                        a.yy02 = [a.yy02[0] + vy2*t]
                        a.xx03 = [a.xx03[0] + vx3*t]
                        a.yy03 = [a.yy03[0] + vy3*t]
                        a.xx04 = [a.xx04[0] + vx4*t]
                        a.yy04 = [a.yy04[0] + vy4*t]
                        a.ssigmax01 = [a.ssigmax01[0] + np.sqrt(2*Dx1*t)]
                        a.ssigmay01 = [a.ssigmay01[0] + np.sqrt(2*Dy1*t)]                  
                        a.ssigmax02 = [a.ssigmax02[0] + np.sqrt(2*Dx2*t)]
                        a.ssigmay02 = [a.ssigmay02[0] + np.sqrt(2*Dy2*t)] 
                        a.ssigmax03 = [a.ssigmax03[0] + np.sqrt(2*Dx3*t)]
                        a.ssigmay03 = [a.ssigmay03[0] + np.sqrt(2*Dy3*t)]                  
                        a.ssigmax04 = [a.ssigmax04[0] + np.sqrt(2*Dx4*t)]
                        a.ssigmay04 = [a.ssigmay04[0] + np.sqrt(2*Dy4*t)] 

                    result = differential_evolution(partial(IniLikelihood1,a),bounds1)  
                    print "result=",result 
                    fitted_params = result.x
                    print(fitted_params)        
                    vx1 = fitted_params[0]
                    vy1 = fitted_params[1]
                    vx2 = fitted_params[2]
                    vy2 = fitted_params[3]
                    a.fitted_params = vx1,vy1,vx2,vy2 
                    
                    #result2 = fmin_tnc(partial(IniLikelihood2,a),x0=a.rr[4:8],approx_grad=True,bounds=bounds1,epsilon=1e-5) 
                    result2 = differential_evolution(partial(IniLikelihood2,a),bounds11) 
                    print "result2=", result2
                    fitted_params2 = result2.x
                    print(fitted_params2)
                    vx3 = fitted_params2[0]
                    vy3 = fitted_params2[1]
                    vx4 = fitted_params2[2]
                    vy4 = fitted_params2[3]
                    a.fitted_params2 = vx3,vy3,vx4,vy4#Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4#,ro1,ro2   
                    #result3 = fmin_tnc(partial(IniLikelihood3,a),x0=a.rr[8:12],approx_grad=True,bounds=bounds2,epsilon=1e-5)
                    result3 = differential_evolution(partial(IniLikelihood3,a),bounds2)
                    print "result3=",result3     
                    # if result3.success:
                    fitted_params3 = result3.x
                    print(fitted_params3)
                    Dx1 = fitted_params3[0]
                    Dy1 = fitted_params3[1]
                    Dx2 = fitted_params3[2]
                    Dy2 = fitted_params3[3]
                    a.fitted_params3 = Dx1,Dy1,Dx2,Dy2
                    print a.fitted_params3  
                    #result4 = fmin_tnc(partial(IniLikelihood4,a),x0=a.rr[12:16],approx_grad=True,bounds=bounds2,epsilon=1e-5)                
                    result4 = differential_evolution(partial(IniLikelihood4,a),bounds33)
                    print "result4=",result4
                    #if result4.success:
                    fitted_params4 = result4.x
                    print(fitted_params4)
                    Dx3 = fitted_params4[0]
                    Dy3 = fitted_params4[1]
                    Dx4 = fitted_params4[2]
                    Dy4 = fitted_params4[3]
                    a.fitted_params4 = Dx3,Dy3,Dx4,Dy4
                    print a.fitted_params4
                    #result5 = fmin_tnc(partial(IniLikelihood5,a),x0=a.rr[16:20],approx_grad=True,bounds=bounds3,epsilon=1e-5)
                    result5 = differential_evolution(partial(IniLikelihood5,a),bounds3)
                    print "result5=",result5    
                    fitted_params5 = result5.x
                    print(fitted_params5)
                    ro1 = fitted_params5[0]
                    ro2 = fitted_params5[1]
                    ro3 = fitted_params5[2]
                    ro4 = fitted_params5[3]
                    a.fitted_params5 = ro1,ro2,ro3,ro4
                    print a.fitted_params5

                    gamma1 = a.r[k][20]
                    gamma2 = a.r[k][21]
                    gamma3 =a.r[k][22]
                    gamma4 = a.r[k][23]
                    par = vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4                    
                    a.par.append(par) 
                np.savetxt('par.txt',a.par,delimiter=',')


        if ourinformation['Run'] == 'Recalc':   
            if ourinformation['Type'] == 'instantaneous':            
                a.par = np.loadtxt('par.txt',delimiter=',')

        if ourinformation['Type'] == 'instantaneous':            
            a.ptind = np.insert(a.t,0,-1)             
            for u in range(len(a.t)):                        
                for i in range(len(pret[u])):
		            a.Pt = a.ptind[u+1]                	
		            t = pret[u]
		            if i >= len(a.par):
		            	vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[-1]  
		            else:                 
		            	vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.par[i]
		            if i == 0:
		                a.xx01 = a.x01 + vx1*t[i]
		                a.yy01 = a.y01 + vy1*t[i]
		                a.xx02 = a.x02 + vx2*t[i]
		                a.yy02 = a.y02 + vy2*t[i]
		                a.xx03 = a.x03 + vx3*t[i]
		                a.yy03 = a.y03 + vy3*t[i]
		                a.xx04 = a.x04 + vx4*t[i]
		                a.yy04 = a.y04 + vy4*t[i]     
		                a.Dx01 = a.DDx01 + np.sqrt(2*Dx1*t[i])
		                a.Dy01 = a.DDy01 + np.sqrt(2*Dy1*t[i])                  
		                a.Dx02 = a.DDx02 + np.sqrt(2*Dx2*t[i])
		                a.Dy02 = a.DDy02 + np.sqrt(2*Dy2*t[i]) 
		                a.Dx03 = a.DDx03 + np.sqrt(2*Dx3*t[i])
		                a.Dy03 = a.DDy03 + np.sqrt(2*Dy3*t[i])                  
		                a.Dx04 = a.DDx04 + np.sqrt(2*Dx4*t[i])
		                a.Dy04 = a.DDy04 + np.sqrt(2*Dy4*t[i])  
		            else:
						a.xx01 = a.xx01 + vx1*t[i]
						a.yy01 = a.yy01 + vy1*t[i]
						a.xx02 = a.xx02 + vx2*t[i]
						a.yy02 = a.yy02 + vy2*t[i]
						a.xx03 = a.xx03 + vx3*t[i]
						a.yy03 = a.yy03 + vy3*t[i]
						a.xx04 = a.xx04 + vx4*t[i]
						a.yy04 = a.yy04 + vy4*t[i] 
						a.Dx01 = a.Dx01 + np.sqrt(2*Dx1*t[i])
						a.Dy01 = a.Dy01 + np.sqrt(2*Dy1*t[i])                  
						a.Dx02 = a.Dx02 + np.sqrt(2*Dx2*t[i])
						a.Dy02 = a.Dy02 + np.sqrt(2*Dy2*t[i]) 
						a.Dx03 = a.Dx03 + np.sqrt(2*Dx3*t[i])
						a.Dy03 = a.Dy03 + np.sqrt(2*Dy3*t[i])                  
						a.Dx04 = a.Dx04 + np.sqrt(2*Dx4*t[i])
						a.Dy04 = a.Dy04 + np.sqrt(2*Dy4*t[i])  
                a.r = a.par[-1]
                lc = zip(x,y)
                #resacf=np.array(integcfcontiins(a,lc[0]))
                resacf=np.array(multicore9(a,lc))
                resacf=np.transpose(resacf)
                rescf.append(resacf)
            scf = np.array(rescf)
            scf = scf/np.max(scf)
            print 'lenscf',len(scf)

        if ourinformation['Run'] == 'Run':   
            if ourinformation['Type'] == 'continuous':
                a.Dx01 = []
                a.Dy01 = []
                a.Dx02 = []
                a.Dy02 = []
                a.Dx03 = []
                a.Dy03 = []
                a.Dx04 = []
                a.Dy04 = []    
                a.par = []    
                STT = []
                a.stind = np.transpose(a.uniST)   
                a.stind = np.insert(a.stind,0,-1)
                print 'prrixx',priorxx           
                for k in range(len(a.DLxx)):
                    a.DLx = a.DLxx[k]
                    a.DLy = a.DLyy[k]           
                    a.DLcon = a.DLccon[k]             
                    a.st = a.sst[k][0]              
                    STI = a.smallspill[a.smallspill>=a.stind[k]]
                    STI = STI[STI<a.stind[k+1]]  # the divided small spill within the sample time                 
                    t=a.stind[k+1]           
                    STT=np.append(STT,STI)  # the old and new time 
                    a.TTi = STT  
                    if ourinformation['SubmergedType']== "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':     
                        a.priorxx = priorxx[k]
                        a.prioryy = prioryy[k]
                        a.priorccon = priorccon[k]
                        a.Datax = DDatax[k]
                        a.Datay = DDatay[k]
                        a.Datacon = DDatacon[k]
                        a.l = td[k]    
                    else: 
                        a.l = 1            

                    a.xx01,a.yy01,a.xx02,a.yy02,a.xx03,a.yy03,a.xx04,a.yy04,a.ssigmax01,a.ssigmay01,a.ssigmax02,a.ssigmay02,a.ssigmax03,a.ssigmay03,a.ssigmax04,a.ssigmay04,a.STT = combinepos[k]
                    a.rr = combinepara[k]
                    print 'a.rr',a.rr                        
                    a.MaxLogLike = MaxLog[k]
                    print 'a.MaxLogLike',a.MaxLogLike

                    result = differential_evolution(partial(IniLikelihood1,a),bounds1)  
                    print "result=",result 
                    fitted_params = result.x
                    print(fitted_params)        
                    vx1 = fitted_params[0]
                    vy1 = fitted_params[1]
                    vx2 = fitted_params[2]
                    vy2 = fitted_params[3]
                    a.fitted_params = vx1,vy1,vx2,vy2 
                    
                    #result2 = fmin_tnc(partial(IniLikelihood2,a),x0=a.rr[4:8],approx_grad=True,bounds=bounds1,epsilon=1e-5) 
                    result2 = differential_evolution(partial(IniLikelihood2,a),bounds11) 
                    print "result2=", result2
                    fitted_params2 = result2.x
                    print(fitted_params2)
                    vx3 = fitted_params2[0]
                    vy3 = fitted_params2[1]
                    vx4 = fitted_params2[2]
                    vy4 = fitted_params2[3]
                    a.fitted_params2 = vx3,vy3,vx4,vy4#Dx1,Dx2,Dx3,Dx4,Dy1,Dy2,Dy3,Dy4#,ro1,ro2   
                    #result3 = fmin_tnc(partial(IniLikelihood3,a),x0=a.rr[8:12],approx_grad=True,bounds=bounds2,epsilon=1e-5)
                    result3 = differential_evolution(partial(IniLikelihood3,a),bounds2)
                    print "result3=",result3     
                    # if result3.success:
                    fitted_params3 = result3.x
                    print(fitted_params3)
                    Dx1 = fitted_params3[0]
                    Dy1 = fitted_params3[1]
                    Dx2 = fitted_params3[2]
                    Dy2 = fitted_params3[3]
                    a.fitted_params3 = Dx1,Dy1,Dx2,Dy2
                    print a.fitted_params3  
                    #result4 = fmin_tnc(partial(IniLikelihood4,a),x0=a.rr[12:16],approx_grad=True,bounds=bounds2,epsilon=1e-5)                
                    result4 = differential_evolution(partial(IniLikelihood4,a),bounds33)
                    print "result4=",result4
                    #if result4.success:
                    fitted_params4 = result4.x
                    print(fitted_params4)
                    Dx3 = fitted_params4[0]
                    Dy3 = fitted_params4[1]
                    Dx4 = fitted_params4[2]
                    Dy4 = fitted_params4[3]
                    a.fitted_params4 = Dx3,Dy3,Dx4,Dy4
                    print a.fitted_params4
                    #result5 = fmin_tnc(partial(IniLikelihood5,a),x0=a.rr[16:20],approx_grad=True,bounds=bounds3,epsilon=1e-5)
                    result5 = differential_evolution(partial(IniLikelihood5,a),bounds3)
                    print "result5=",result5    
                    fitted_params5 = result5.x
                    print(fitted_params5)
                    ro1 = fitted_params5[0]
                    ro2 = fitted_params5[1]
                    ro3 = fitted_params5[2]
                    ro4 = fitted_params5[3]
                    a.fitted_params5 = ro1,ro2,ro3,ro4
                    print a.fitted_params5
                    print 'a.rr',a.rr
                    print 'a.rr',a.rr[k]
                    gamma1 = a.rr[-4]
                    gamma2 = a.rr[-3]
                    gamma3 =a.rr[-2]
                    gamma4 = a.rr[-1]
                    par = vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4                    
                    a.par.append(par)  
                np.savetxt('par.txt',a.par,delimiter=',')          

        if ourinformation['Run'] == 'Recalc':   
            if ourinformation['Type'] == 'continuous':            
                a.par = np.loadtxt('par.txt',delimiter=',')
            print 'a.ppppppar',a.par
        if ourinformation['Type'] == 'continuous':     
            a.ptind = np.insert(a.t,0,-1)             
            for u in range(len(a.t)):
                a.Pt = a.ptind[u+1]
                diff = a.Pt - a.stind
                index = np.where(diff>=0)
                if a.Pt >= np.max(a.stind):
                    if ourinformation['Run'] == 'Run':  
                        a.xx1,a.yy1,a.xx2,a.yy2,a.xx3,a.yy3,a.xx4,a.yy4,a.sx1,a.sy1,a.sx2,a.sy2,a.sx3,a.sy3,a.sx4,a.sy4,STT = postcombinepara[-1]    
                    if ourinformation['Run'] == 'Recalc':    
                        a.xx1,a.yy1,a.xx2,a.yy2,a.xx3,a.yy3,a.xx4,a.yy4,a.sx1,a.sy1,a.sx2,a.sy2,a.sx3,a.sy3,a.sx4,a.sy4,STT = postcombinepara    

                    a.r = a.par[-1]  # parameter
                    print 'a.rrrrrr',a.r 
                    vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.r
                    PTI = a.smallspill[a.smallspill>=STT[-1]]
                    PTI = PTI[PTI<a.Pt]  # the divided small spill within the sample time                 
                    STT = np.append(STT,PTI) 
                    a.ps = STT               
                    a.xx1=np.append(a.xx1,[a.x01]*len(PTI))
                    a.xx2=np.append(a.xx2,[a.x02]*len(PTI))   
                    a.xx3=np.append(a.xx3,[a.x03]*len(PTI))
                    a.xx4=np.append(a.xx4,[a.x04]*len(PTI))                     
                    a.yy1=np.append(a.yy1,[a.y01]*len(PTI))
                    a.yy2=np.append(a.yy2,[a.y02]*len(PTI))   
                    a.yy3=np.append(a.yy3,[a.y03]*len(PTI))
                    a.yy4=np.append(a.yy4,[a.y04]*len(PTI))                    
                    a.sx1=np.append(a.sx1,[a.sigmax01]*len(PTI))
                    a.sy1=np.append(a.sy1,[a.sigmay01]*len(PTI))   
                    a.sx2=np.append(a.sx2,[a.sigmax02]*len(PTI))
                    a.sy2=np.append(a.sy2,[a.sigmay02]*len(PTI))                     
                    a.sx3=np.append(a.sx3,[a.sigmax03]*len(PTI))
                    a.sy3=np.append(a.sy3,[a.sigmay03]*len(PTI))   
                    a.sx4=np.append(a.sx4,[a.sigmax04]*len(PTI))
                    a.sy4=np.append(a.sy4,[a.sigmay04]*len(PTI))

                    a.xx1 = a.xx1 + vx1*(a.Pt-STT)
                    a.yy1 = a.yy1 + vy1*(a.Pt-STT)   
                    a.xx2 = a.xx2 + vx2*(a.Pt-STT)
                    a.yy2 = a.yy2 + vy2*(a.Pt-STT)    
                    a.xx3 = a.xx3 + vx3*(a.Pt-STT)
                    a.yy3 = a.yy3 + vy3*(a.Pt-STT)   
                    a.xx4 = a.xx4 + vx4*(a.Pt-STT)
                    a.yy4 = a.yy4 + vy4*(a.Pt-STT)  

                    a.sx1 = a.sx1 + np.sqrt(2*Dx1*(a.Pt-STT))
                    a.sy1 = a.sy1 + np.sqrt(2*Dy1*(a.Pt-STT))                  
                    a.sx2 = a.sx2 + np.sqrt(2*Dx2*(a.Pt-STT))
                    a.sy2 = a.sy2 + np.sqrt(2*Dy2*(a.Pt-STT)) 
                    a.sx3 = a.sx3 + np.sqrt(2*Dx3*(a.Pt-STT))
                    a.sy3 = a.sy3 + np.sqrt(2*Dy3*(a.Pt-STT))                  
                    a.sx4 = a.sx4 + np.sqrt(2*Dx4*(a.Pt-STT))
                    a.sy4 = a.sy4 + np.sqrt(2*Dy4*(a.Pt-STT))                
                lc = zip(x,y)
                #resacf=np.array(integcfconti(a,lc[0]))                
                resacf=np.array(multicore6(a,lc))
                resacf=np.transpose(resacf)
                rescf.append(resacf)
            scf = np.array(rescf)
            scf = scf/np.max(scf)
            print 'lenscf',len(scf)            

    elif ourinformation['Method'] == 'Best':
        pass 
    #_______________confidence bounds_____________
    if ourinformation['Type'] == 'continuous':
        TT = []
        a.xx1=[]
        a.yy1=[]
        a.xx2=[]
        a.yy2=[]
        a.xx3=[]
        a.yy3=[]
        a.xx4=[]
        a.yy4=[]    
        a.sx1 = []
        a.sx2 = []
        a.sx3 = []
        a.sx4 = []
        a.sy1 = []
        a.sy2 = []
        a.sy3 = []
        a.sy4 = []        
        res = []
        loc = zip(x,y)        
        a.stind = np.transpose(a.uniST)  # sample time  
        a.ptind = np.insert(a.t,0,-1)          
        for u in range(len(a.t)):
            a.Pt = a.ptind[u+1]
            diff = a.Pt - a.stind
            index = np.where(diff>=0)
            if a.Pt < np.max(a.stind):
                a.xx1,a.yy1,a.xx2,a.yy2,a.xx3,a.yy3,a.xx4,a.yy4,a.sx1,a.sy1,a.sx2,a.sy2,a.sx3,a.sy3,a.sx4,a.sy4,STT = combinepos[np.max(index)]             
                a.r = combinepara[np.max(index)]  # parameter
                a.ps = STT
            if a.Pt >= np.max(a.stind):
                if ourinformation['Run'] == 'Run':    
                    a.xx1,a.yy1,a.xx2,a.yy2,a.xx3,a.yy3,a.xx4,a.yy4,a.sx1,a.sy1,a.sx2,a.sy2,a.sx3,a.sy3,a.sx4,a.sy4,STT = postcombinepara[-1]     
                if ourinformation['Run'] == 'Recalc':    
                    a.xx1,a.yy1,a.xx2,a.yy2,a.xx3,a.yy3,a.xx4,a.yy4,a.sx1,a.sy1,a.sx2,a.sy2,a.sx3,a.sy3,a.sx4,a.sy4,STT = postcombinepara     
                a.r = combinepara[-1]  # parameter
                vx1,vy1,vx2,vy2,vx3,vy3,vx4,vy4,Dx1,Dy1,Dx2,Dy2,Dx3,Dy3,Dx4,Dy4,ro1,ro2,ro3,ro4,gamma1,gamma2,gamma3,gamma4 = a.r
                PTI = a.smallspill[a.smallspill>=STT[-1]]
                PTI = PTI[PTI<a.Pt]  # the divided small spill within the sample time                 
                STT = np.append(STT,PTI) 
                a.ps = STT               
                a.xx1=np.append(a.xx1,[a.x01]*len(PTI))
                a.xx2=np.append(a.xx2,[a.x02]*len(PTI))   
                a.xx3=np.append(a.xx3,[a.x03]*len(PTI))
                a.xx4=np.append(a.xx4,[a.x04]*len(PTI))                     
                a.yy1=np.append(a.yy1,[a.y01]*len(PTI))
                a.yy2=np.append(a.yy2,[a.y02]*len(PTI))   
                a.yy3=np.append(a.yy3,[a.y03]*len(PTI))
                a.yy4=np.append(a.yy4,[a.y04]*len(PTI))                    
                a.sx1=np.append(a.sx1,[a.sigmax01]*len(PTI))
                a.sy1=np.append(a.sy1,[a.sigmay01]*len(PTI))   
                a.sx2=np.append(a.sx2,[a.sigmax02]*len(PTI))
                a.sy2=np.append(a.sy2,[a.sigmay02]*len(PTI))                     
                a.sx3=np.append(a.sx3,[a.sigmax03]*len(PTI))
                a.sy3=np.append(a.sy3,[a.sigmay03]*len(PTI))   
                a.sx4=np.append(a.sx4,[a.sigmax04]*len(PTI))
                a.sy4=np.append(a.sy4,[a.sigmay04]*len(PTI))

                a.xx1 = a.xx1 + vx1*(a.Pt-STT)
                a.yy1 = a.yy1 + vy1*(a.Pt-STT)   
                a.xx2 = a.xx2 + vx2*(a.Pt-STT)
                a.yy2 = a.yy2 + vy2*(a.Pt-STT)    
                a.xx3 = a.xx3 + vx3*(a.Pt-STT)
                a.yy3 = a.yy3 + vy3*(a.Pt-STT)   
                a.xx4 = a.xx4 + vx4*(a.Pt-STT)
                a.yy4 = a.yy4 + vy4*(a.Pt-STT)  

                a.sx1 = a.sx1 + np.sqrt(2*Dx1*(a.Pt-STT))
                a.sy1 = a.sy1 + np.sqrt(2*Dy1*(a.Pt-STT))                  
                a.sx2 = a.sx2 + np.sqrt(2*Dx2*(a.Pt-STT))
                a.sy2 = a.sy2 + np.sqrt(2*Dy2*(a.Pt-STT)) 
                a.sx3 = a.sx3 + np.sqrt(2*Dx3*(a.Pt-STT))
                a.sy3 = a.sy3 + np.sqrt(2*Dy3*(a.Pt-STT))                  
                a.sx4 = a.sx4 + np.sqrt(2*Dx4*(a.Pt-STT))
                a.sy4 = a.sy4 + np.sqrt(2*Dy4*(a.Pt-STT))                                
            #resa = integcontinuous(a,loc)
            resa = np.array(multicore5(a,loc))
            resa = np.transpose(resa)
            res.append(resa)
        res = np.array(res)
        s = res/np.max(res) 
    #print "s=",s 
#_________display in km ___________________
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

    #km prior data
    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
        newxprior = np.zeros([len(a.DLlonprior)])
        for i in range(len(a.DLlonprior)):
            newxprior[i] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],a.DLlonprior[i])).km 

        newyprior = np.zeros([len(a.DLlatprior)])
        for i in range(len(a.DLlatprior)):
            newyprior[i] = geopy.distance.distance((xkm[0],ykm[0]),(a.DLlatprior[i],ykm[0])).km 

    #km lat0 lon0 
    lon0km = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],lon0)).km 
    lat0km = geopy.distance.distance((xkm[0],ykm[0]),(lat0,ykm[0])).km 

    if ourinformation['spillname'] == 'DWH':
        pltpxkm = []
        ppltpxkm = []
        for i in range(len(ppltpx)):
            for j in range(len(ppltpx[i])):
                pltpxk = np.ones([len(ppltpx[i][j])])
                for k in range(len(ppltpx[i][j])): 
                    pltpxk[k] = geopy.distance.distance((xkm[0],ykm[0]),(xkm[0],ppltpx[i][j][k])).km 
                pltpxkm.append(pltpxk)
            ppltpxkm.append(pltpxkm)

        pltpykm = []
        ppltpykm = []
        for i in range(len(ppltpy)):
            for j in range(len(ppltpy[i])):        
                pltpyk = np.ones([len(ppltpy[i][j])])
                for k in range(len(ppltpy[i][j])): 
                    pltpyk[k] = geopy.distance.distance((xkm[0],ykm[0]),(ppltpy[i][j][k],ykm[0])).km 
                pltpykm.append(pltpyk)
            ppltpykm.append(pltpykm)
        print 'pltppppk',ppltpxkm
    else:
        xaa = xaxis
        yaa = yaxis
#_______________km___________________________________________________
# plot in km 
    labels = []
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    labels5 = []
    labels6 = []  
    labels7 = []    
    labelsc = []       
    for tt in range(len(a.t)):
        print 'tt',tt
        plt.figure()
        plt.rcParams['font.size'] = 10   # change the font size of colorbar
        if ourinformation['Method'] == 'Best':
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))            
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' # make the test bolder                
                plt.colorbar()                                
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'                 
                cs3=plt.plot(a.lon0,a.lat0,ms=9,c='r',marker='+',label='Spill_Location')               
                plt.legend([cs3],['Spill_Location'], loc='upper left',ncol=3, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                   
                if ourinformation['Plot'] =="nofield":
                    pass
                elif ourinformation['Plot']=="field":
                    plt.scatter(a.DLlonfield,a.DLlatfield,s=a.DLconfield,label="Field_Data",c='b')                  
                    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                                         
                    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                        lon = a.DLlonprior
                        lat = a.DLlatprior
                        con = a.DLconprior
                        plt.scatter(lon,lat,s=con,label="Prior_Data",color='darkred')  
                        plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))     
                    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                                                                                              
                if ourinformation['contour'] =="nocontour":
                    pass                    
                elif ourinformation['contour'] == 'contour':
                    if ourinformation['spillname'] == 'DWH':
                        cs = plt.contour(ppltpx[tt],ppltpy[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=1.0)
                    plt.xlim(a.lon0-a.scale[0],a.lon0+a.scale[0])
                    plt.ylim(a.lat0-a.scale[1],a.lat0+a.scale[1])
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    level=cs.levels
                    print 'level',level
                    for i in range(len(level)):
                    	if i == 0: 
                            lab = 'Iso Upper Depths'
                            labels.append(lab)
                    for i in range(len(labels)):
                    	if i == 0:                         	
                            cs.collections[i].set_label(labels[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))  
                    if ourinformation['spillname'] == 'DWH':                    
                        cs2 = plt.contour(pltpx,pltpy,pltZ1,levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else: 
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=1.0)                                                  
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul
                    level=cs2.levels
                    print 'level2',level
                    for i in range(len(level)):
                    	if i == 0:                         	
                            lab = 'Iso Lower Depths'
                            labels1.append(lab)
                    print 'labels',labels1
                    for i in range(len(labels1)):
                    	if i == 0:                         	
                            cs2.collections[i].set_label(labels1[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))    
                     
            elif ourinformation['Map'] == 'km': 
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(Xa),len(Ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 13   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' # make the test bolder                  
                plt.colorbar()
                plt.rcParams['font.size'] = 9   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' 
                cs3=plt.plot(lon0km,lat0km,ms=9,c='r',marker='+',label='Spill_Location') # change !!!!location                                       
                plt.legend([cs3],['Spill_Location'], loc='upper left',ncol=3, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))       

                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=a.DLconfield,label="Field_Data",c='b')
                    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))     
                    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                        lon = newxprior
                        lat = newyprior
                        con = a.DLconprior
                        plt.scatter(lon,lat,s=con,label="Prior_Data",color='darkred')  
                        plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))  

                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour":  
                    if ourinformation['spillname'] == 'DWH':                                  
                        cs = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                        
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    level=cs.levels
                    print 'level',level
                    for i in range(len(level)):
                        if i == 0:                     	
	                        lab = 'Iso Upper Depths'
	                        labels.append(lab)
                    for i in range(len(labels)):
                    	if i == 0: 
	                        cs.collections[i].set_label(labels[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))  
                    if ourinformation['spillname'] == 'DWH':                                                      
                        cs2 = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ1[tt],levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                          
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul
                    level=cs2.levels
                    for i in range(len(level)):
                        if i == 0:                          
                            lab = 'Iso Lower Depths'
                            labels1.append(lab)
                    print 'labels',labels1
                    for i in range(len(labels1)):
                        if i == 0:                          
                            cs2.collections[i].set_label(labels1[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))                           

        elif ourinformation['Method'] == 'Minimum':                
            if ourinformation['Map'] == 'Coordinate':
                plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' # make the test bolder                   
                plt.colorbar()  
                plt.rcParams['font.size'] = 10   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'      
                cs3=plt.plot(a.lon0,a.lat0,ms=9,c='r',marker='+',label='Spill_Location') # change !!!!location                                                   
                cs4=plt.contour(Ya,Xa,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g'])    
                
                level=cs4.levels                
                for i in range(len(level)):
                    lab = 'Conf. Bound'+str(level[i])
                    labelsc.append(lab)   
                for i in range(len(labelsc)):                                 
                    cs4.collections[i].set_label(labelsc[i])
                plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))      
      ######add the minimum regret part           
                if ourinformation['Plot'] == 'field':
                    plt.scatter(a.DLlonfield,a.DLlatfield,s=a.DLconfield,label='Field_Data')  
                    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                     
                    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                        if len(a.DLlonprior)>0:
                            lon = a.DLlonprior
                            lat = a.DLlatprior
                            con = a.DLconprior
                            plt.scatter(lon,lat,s=con,label='Prior_Data',color='darkred')
                            plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                     

                elif ourinformation['Plot'] =="nofield":  
                    pass
                if ourinformation['contour'] =='contour':
                    if ourinformation['spillname'] == 'DWH':                                                      
                        cs = plt.contour(ppltpx[tt],ppltpy[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                
                    plt.xlim(a.lon0-a.scale[0],a.lon0+a.scale[0])
                    plt.ylim(a.lat0-a.scale[1],a.lat0+a.scale[1])
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    level=cs.levels
                    print 'level',level
                    for i in range(len(level)):
                        if i == 0:                         
                            lab = 'Iso Upper Depths'
                            labels.append(lab)
                    for i in range(len(labels)):
                        if i == 0:                         
                            cs.collections[i].set_label(labels[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))  
                    if ourinformation['spillname'] == 'DWH':                                                                          
                        cs2 = plt.contour(ppltpx[tt],ppltpy[tt],ppltZ1[tt],levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                  
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul
                    level=cs2.levels
                    print 'level2',level
                    for i in range(len(level)):
                        if i == 0:                         
                            lab = 'Iso Lower Depths'
                            labels1.append(lab)
                    print 'labels',labels1
                    for i in range(len(labels1)):
                        if i == 0:                         
                            cs2.collections[i].set_label(labels1[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))                                                  
                elif ourinformation['contour'] =="nocontour":
                    pass
            elif ourinformation['Map'] == 'km':
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(Xa),len(Ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.rcParams['font.size'] = 13   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold' # make the test bolder                           
                plt.colorbar()       
                plt.rcParams['font.size'] = 9   # change the font size of colorbar
                plt.rcParams['font.weight'] = 'bold'  
                cs3=plt.plot(lon0km,lat0km,ms=9,c='r',marker='+',label='Spill_Location') # change !!!!location                                       
                plt.legend([cs3],['Spill_Location'], loc='upper left',ncol=3, mode="expand", borderaxespad=0.,facecolor='white',frameon=True,edgecolor='white',fontsize='small')                                             
                cs4=plt.contour(xaxis,yaxis,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g']) 
                level=cs4.levels                
                
                for i in range(len(level)):
                    lab = 'Conf. Bound'+str(level[i])
                    labelsc.append(lab)   
                for i in range(len(labelsc)):                                 
                    cs4.collections[i].set_label(labelsc[i])

                plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))
                if ourinformation['Plot'] =="nofield":  
                    pass
                elif ourinformation['Plot'] =="field":  
                    plt.scatter(SXkm,SYkm,s=a.DLconfield,c='b')
                    plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))
                    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                        lon = newxprior
                        lat = newyprior
                        con = a.DLconprior
                        plt.scatter(lon,lat,s=con,label="Prior_Data",color='darkred')  
                        plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                      
                if ourinformation['contour'] =="nocontour":
                    pass
                elif ourinformation['contour'] =="contour": 
                    if ourinformation['spillname'] == 'DWH':                                                                          
                        cs = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                                        
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    level=cs.levels
                    print 'level',level
                    for i in range(len(level)):
                        if i == 0:                         
                            lab = 'Iso Upper Depths'
                            labels.append(lab)
                    for i in range(len(labels)):
                        if i == 0:                         
                            cs.collections[i].set_label(labels[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))  
                    if ourinformation['spillname'] == 'DWH':                                                                          
                        cs2 = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ1[tt],levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                                          
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul
                    level=cs2.levels
                    print 'level2',level
                    for i in range(len(level)):
                        if i == 0:                         
                            lab = 'Iso Lower Depths'
                            labels1.append(lab)
                    print 'labels',labels1
                    for i in range(len(labels1)):
                        if i == 0:                         
                            cs2.collections[i].set_label(labels1[i])                        
                    plt.legend(loc='upper left',facecolor='white',edgecolor='white',frameon=True,fontsize='medium',ncol=3,bbox_to_anchor=(0,1.02,1,0.2))      

        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        fffilename  = "Results/submerged"+time+"_back.png"
        plt.savefig(fffilename,bbox_inches="tight",dpi=2000)
        plt.clf()
        img = cv2.imread(fffilename)        
        h, w,_= img.shape
        crop_img = img[int(h*0.1):h,int(w*0.72):w]  # legend location
        crop_img2 = img[0:int(h/7),0:int(w)]           
        fffilename1 = "Results/submerged"+time+"_legend.png"
        cv2.imwrite(fffilename1,crop_img)
        fffilename2 = "Results/submerged"+time+"_outputlegend.png"        
        cv2.imwrite(fffilename2,crop_img2)        
        plt.clf()

# 	print "_______________________________Above is True_________________________________________"
	if ourinformation['Method'] == 'Best':                     
            if ourinformation['Map'] == 'Coordinate':    
                plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))                
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
                        DLlon = a.DLlonfield
                        DLlat = a.DLlatfield
                        DLcon = a.DLconfield
                        plt.scatter(DLlon,DLlat,s=DLcon,c='b')  
                        if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                            if len(a.DLlonprior)>0:
                                lon = a.DLlonprior
                                lat = a.DLlatprior
                                con = a.DLconprior
                                plt.scatter(lon,lat,s=con,color='darkred')  
                            else: 
                                pass
	            
                elif ourinformation['Plot'] == 'nofield':
	                pass
                if ourinformation['contour'] =='contour':
                    if ourinformation['spillname'] == 'DWH':                                                                                              
                        cs = plt.contour(ppltpx[tt],ppltpy[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                                        
                    
                    plt.xlim(a.lon0-a.scale[0],a.lon0+a.scale[0])
                    plt.ylim(a.lat0-a.scale[1],a.lat0+a.scale[1])
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)  
                    if ourinformation['spillname'] == 'DWH':                                                                                              
                        cs2 = plt.contour(pltpx,pltpy,pltZ1,levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                                                                  
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul                   
                elif ourinformation['contour'] =='nocontour':
	                pass

            if ourinformation['Map'] == 'km': 
                    print '__________________________________'
	            plt.figure()
                    plt.contourf(xaxis,yaxis,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
                    print 'xaxis',xaxis              
                    plt.box(on=0)
                    plt.grid(color = 'w', linestyle='-', linewidth=1)
                    plt.plot(lon0km,lat0km,ms=10,c='r')
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
    	                plt.scatter(SXkm,SYkm,s=a.DLconfield,c="b",label="Field Data")#,marker=r'$\clubsuit$')
    	                print "I am field"             
                        if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                            lon = newxprior
                            lat = newyprior
                            con = a.DLconprior
                            plt.scatter(lon,lat,s=con,label="Prior_Data",color='darkred')  
                            plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))  
                    elif ourinformation['Plot'] == 'nofield':
    	                print "nofield"
                    if ourinformation['contour'] =='contour': 
                        print '------------------------'
                        if ourinformation['spillname'] == 'DWH':                                                                                              
                            cs = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                        else:
                            cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                                        
                        plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)   
                        if ourinformation['spillname'] == 'DWH':                                                                                              
                            cs2 = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ1[tt],levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                        else:
                            cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                                                                                              
                        plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul  

	elif ourinformation['Method'] == 'Minimum':
	    if ourinformation['Map'] == 'Coordinate':    #### add minimum part 
	        plt.contourf(Ya,Xa,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
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
                    DLlon = a.DLlonfield
                    DLlat = a.DLlatfield
                    DLcon = a.DLconfield
                    plt.scatter(DLlon,DLlat,s=DLcon,c='b')  
                    if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                        if len(a.DLlonprior)>0:
                            lon = a.DLlonprior
                            lat = a.DLlatprior
                            con = a.DLconprior
                            plt.scatter(lon,lat,s=con,color='darkred')  
                          
                elif ourinformation['Plot'] == 'nofield':
	                print "nofield"
                if ourinformation['contour'] =='contour':
                    if ourinformation['spillname'] == 'DWH':                                                                                              
                        cs = plt.contour(ppltpx[tt],ppltpy[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else: 
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                                        
                    plt.xlim(a.lon0-a.scale[0],a.lon0+a.scale[0])
                    plt.ylim(a.lat0-a.scale[1],a.lat0+a.scale[1])
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)   
                    if ourinformation['spillname'] == 'DWH':                                                                                                                  
                        cs2 = plt.contour(pltpx,pltpy,pltZ1,levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                                                                                              
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul  
                elif ourinformation['contour'] =='nocontour':
	                pass

	    elif ourinformation['Map'] == 'km': 
                plt.figure()            
                plt.contourf(xaxis,yaxis,s[tt].reshape(len(xa),len(ya)),levels=np.around(np.linspace(0,1,25),decimals=2),cmap=plt.get_cmap('plasma'))
                plt.contour(xaxis,yaxis,scf[tt].reshape(len(xa),len(ya)), levels=[float(ourinformation["level"])], colors=['g'])   
                plt.plot(lon0km,lat0km,ms=10,c='r')
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
	                plt.scatter(SXkm,SYkm,s=a.DLconfield,c="b",label="Field Data")#,marker=r'$\clubsuit$')   
                        if ourinformation['SubmergedType'] == "OSCAR" or ourinformation['SubmergedType']== "GNOME" or ourinformation['SubmergedType'] == 'Other':    
                            lon = newxprior
                            lat = newyprior
                            con = a.DLconprior
                            plt.scatter(lon,lat,s=con,label="Prior_Data",color='darkred')  
                            plt.legend(scatterpoints=1, frameon=True, labelspacing=1,ncol=3,facecolor='white',edgecolor='white',fontsize='small',bbox_to_anchor=(0,1.02,1,0.2))                            
                elif ourinformation['Plot'] == 'nofield':
	                print "nofield"
                if ourinformation['contour'] =='contour':
                    if ourinformation['spillname'] == 'DWH':                                                                                                                  
                        cs = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ[tt],levels=np.around(np.linspace(np.nanmin(pltZ),np.nanmax(pltZ),4)),colors='y',linestyles='dashed',linewidths=0.5)                       
                    else:
                        cs = plt.contour(yaa,xaa,np.array(zminf).reshape(len(yaa),len(xaa)),levels=np.around(np.linspace(800,np.max(zminf)+(np.max(zminf)-800)/1,3)),colors='y',linestyles='dashed',linewidths=0.5)                                                                                                
                    plt.clabel(cs,incline=1,fontsize=10,weight='bold')#,manual=manual_locations)   
                    if ourinformation['spillname'] == 'DWH':                                                                                                                                      
                        cs2 = plt.contour(ppltpxkm[tt],ppltpykm[tt],ppltZ1[tt],levels=np.around(np.linspace(np.nanmin(pltZ1),np.nanmax(pltZ1),4)),colors='k',linestyles='dashed',linewidths=0.5)                                                                   
                    else:
                        cs2 = plt.contour(yaa,xaa,np.array(zmaxf).reshape(len(xaa),len(yaa)),levels=np.around(np.linspace(np.max(zminf),np.max(zmaxf)+(np.max(zmaxf)-np.max(zminf))/3,5)),colors='k',linestyles='dashed',linewidths=0.5)                                                                                                                                                                                                      
                    plt.clabel(cs2,incline=1,fontsize=10,weight='bold')#,manual=manul     

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        time = datetime.datetime.now()
        time = time.strftime("%Y-%m-%d %H-%M-%S")
        filename  = "Results/submerged"+time+".png"
        #plt.show()
        plt.savefig(filename, dpi=599, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False)
        figPGW = np.array([(float(ourinformation['x_max']) - float(ourinformation['x_min']))/4792.0, 0.0, 0.0, -(float(ourinformation['y_max']) - float(ourinformation['y_min']))/3600.0, float(ourinformation['x_min']), float(ourinformation['y_max'])])
        print 'Xticks',Xticks
        figPGW = np.array([(float(Xticks[len(Xticks)-1]) - float(Xticks[0]))/4792.0, 0.0, 0.0, -(float(Yticks[len(Yticks)-1]) - float(Yticks[0]))/3600.0, float(Xticks[0]), float(Yticks[len(Yticks)-1])])
        filename11 = "Results/submerged"+time+".pgw"
        figPGW.tofile(filename11 , sep = '\n', format = "%.16f")

        img = cv2.imread(filename,1)
        num_rows, num_cols = img.shape[:2]
        if ourinformation['Map'] == 'Coordinate':
            translation_matrix = np.float32([ [1,0,635], [0,1,240] ])
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 635, num_rows + 240))        
        if ourinformation['Map'] == 'km': 
            translation_matrix = np.float32([ [1,0,0], [0,1,0] ])
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
            img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))         
        img_translation[np.where((img_translation == [0,0,0]).all(axis = 2))] = [135,0,0]   
        cv2.imwrite("Results/submerged"+time+".png",img_translation)
        filename1 = "Results/submerged"+time+".png"         
        return filename,fffilename1,fffilename2

# if __name__ == "__main__":
#     myinformation = {'spillname':'DBL','Plot': 'field','Run':'Recalc',"confidence": 0.95,"level":0.01,'Type': 'continuous','starttime':u'2010-04-20 00:00:00','endtime':u'2010-07-15 00:00:00','vymax': 3.0, 'Ratio': u'1', 'PredictTime': [u'2010-05-26 00:00:00'],'OurTime': [u'2010-05-26 00:00:00'], 'CampaignButton': [u'C:/Users/Administrator/OneDrive - University of Miami/Attachments/SOSim2.0/SOSim3.0/SOSim2.0/SOSimfinal_April/New folder/DWH_test.csv'], 'contour': 'contour', 'y_min': 28.6391, 'Map': 'km', 'lonscale': 0.1, 'Run': 'Run', 'vxmax': 3.0, 'vxmin': -3.0, 'lon': -88.3664, 'Method': 'Best', 'SubmergedType':'Nodate', 'latscale': 0.1, 'HydroButton': u'C:/Users/Administrator/OneDrive - University of Miami/Attachments/SOSim2.0/SOSim3.0/SOSim2.0/SOSimfinal_April/New folder/prior22.csv', 'starttime': u'2010-04-20 00:00:00', 'dxmax': 0.89, 'yNode': 25, 'x_min': -88.47794639175258, 'x_max': -88.25485360824742, 'lat': 28.7391, 'endtime': u'2010-07-15 00:00:00', 'dxmin': 0.01, 'y_max': 28.839100000000002, 'dymin': 0.01, 'dymax': 0.89, 'xNode': 25, 'vymin': -3.0, 'OilType': 5}
#     submerged_main(myinformation)
#     #myinformation = {'Plot': 'field', 'Type': 'instantaneous','vymax': 3.0, 'OurTime': [u'2010-04-20 00:00:00'], 'PredictTime': [u'2010-05-16 00:00:00'],'CampaignButton':[u'C:/Users/Administrator/OneDrive - University of Miami/Attachments/SOSim2.0/SOSim3.0/SOSim2.0/SOSim/DWH_515.csv'], 'Method': 'Best', 'contour': 'nocontour', 'y_min': 28.7, 'Map': 'Coordinate', 'lonscale': u'0.4', 'vxmax': 3.0, 'vxmin': -3.0, 'lon': -88.36594, 'instantaneous': 1, 'SubmergedType': 'OSCAR', 'latscale': u'0.3', 'starttime': u'2010-04-20 00:00:00', 'dxmax': 0.89, 'yNode': 40, 'x_min': -88.5, 'x_max': -88.0, 'lat': 28.7381, 'endtime': u'2010-07-15 00:00:00', 'dxmin': 0.01, 'y_max': 29.2, 'dymin': 0.01, 'dymax': 0.89, 'xNode': 40, 'vymin': -3.0, 'OilType': 5}
#     myinformation = {'Plot': 'field',"confidence": 0.95, 'vymax': 3.0, "level":0.01,'Ratio': u'1','Type': 'continuous','OurTime': [u'2010-04-20 00:00:00'], 'PredictTime': [u'2010-05-26 00:00:00'],'CampaignButton': [u'C:/Users/Administrator/OneDrive - University of Miami/Attachments/SOSim2.0/SOSim3.0/SOSim2.0/SOSimfinal_April/New folder/DWH_test.csv'], 'Method': 'Minimum', 'contour': 'contour', 'y_min': 28.638099999999998, 'Map': 'Coordinate', 'lonscale': 0.1, 'Run': 'Run', 'vxmax': 3.0, 'vxmin': -3.0, 'lon': -88.36694, 'instantaneous': 1, 'SubmergedType': 'OSCAR', 'latscale': 0.1, 'HydroButton': u'C:/Users/Administrator/OneDrive - University of Miami/Attachments/SOSim2.0/SOSim3.0/SOSim2.0/SOSimfinal_April/New folder/DWHMay.nc', 'starttime': u'2010-04-20 00:00:00', 'dxmax': 0.89, 'yNode': 25, 'x_min': -88.47848639175258, 'x_max': -88.25539360824742, 'lat': 28.7381, 'endtime': u'2010-04-20 00:00:00', 'dxmin': 0.01, 'y_max': 28.8381, 'dymin': 0.01, 'dymax': 0.89, 'xNode': 25, 'vymin': -3.0, 'OilType': 5}

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
