# -*- coding: utf-8 -*-

#"A predictive Bayesian data-derived multi-modal Gaussian model of sunken oil mass" Article by Angelica Echavarria-Gregory.
#Angelica's dissertation "Predictive Data-Derived Bayesian Statistic-Transport Model and Simulator of Sunken Oil Mass" can also be used as a reference.
#Reading the article and dissertation can provide you with an understanding of the background statistics and analysis.

from __future__ import division
import sys
sys.path.append('/Users/maryjacketti/Desktop/SOSim/SOSim')


import itertools
from math import *
import numpy as np
import random
import mcint
import utm
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import pool
from SOSimCoreC import *
from SOSimOPIC import *
import SOSimCoreC as SOSimCore
import SOSimOPIC as SOSimOPI


import datetime

#This section describes the conditional bivariate Gaussian distribution.
#This is the conditional sampling distribution for the 2-D Bayesian model.
#mux and muy are the 2-D means, sigmax and sigmay are the 2-D covariance matrixes
#DLx and DLy are the difference between the points to be modeled and the spill points
#rho is the correlation coefficient
#ff is used to define mux,muy,sigmax,sigmay and returns the value for the conditional bivariate Gaussian distribution
def B_sampling(DLx,DLy,mux,muy,sigmax,sigmay,rho): #definition of Bm in the conditional probability equation
    Bs=((((DLx-mux))**(2.0))/((sigmax)**2.0))+((((DLy-muy))**(2.0))/((sigmay)**2.0))-((2.0*(rho)*(DLx-mux)*(DLy-muy))/(sigmax*sigmay))
    return Bs 

def CG(sx,sy,BuSamp,ro): #conditional bivariate Gaussian distribution
    CG=((1.0)/(2.0*np.pi*sx*sy*(sqrt(1-((ro)**2.0)))))*(exp(-(BuSamp)/(2.0*(1.0-((ro)**2.0)))))
    return CG

def ff(x,y,mux,muy,sigmax,sigmay,rho,x0,y0,t,sigmax0,sigmay0): #defines the gaussian function
    mux = x0 + mux*t #characterizes initial velocity
    muy = y0 + muy*t 
    sigmax = sigmax0 + np.sqrt(2.0*sigmax*t) #characterizes initial diffusion coefficient
    sigmay= sigmay0 + np.sqrt(2.0*sigmay*t)
    return CG(sigmax,sigmay,B_sampling(x,y,mux,muy,sigmax,sigmay,rho),rho)*1000


#This section describes the first component of the Bayesian analytical model (referenced in Angelica's paper).
#This sums the multiplication of the conditional bivariate Gaussian distribution with the gamma's for the 4 patches.
#The breakdown of the integration was separated into 2 parts: first, the gamma and conditional Bivariate would be multiplied
#second, the likelihood function would be determined and multiplied by the first part.
def f(x,y,gamma,mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4,x0,y0,t,sigmax0,sigmay0): #defines the sum of the four patches multiplied by gamma (first part in Bayesian equation)
    return gamma[0]*ff(x,y,mux1,muy1,sigmax1,sigmay1,rho1,x0,y0,t,sigmax0,sigmay0) + gamma[1]*ff(x,y,mux2,muy2,sigmax2,sigmay2,rho2,x0,y0,t,sigmax0,sigmay0) + gamma[2]*ff(x,y,mux3,muy3,sigmax3,sigmay3,rho3,x0,y0,t,sigmax0,sigmay0) + gamma[3]*ff(x,y,mux4,muy4,sigmax4,sigmay4,rho4,x0,y0,t,sigmax0,sigmay0)

def forwd(x,y,gamma,var,x0,y0,t,sigmax0,sigmay0): #assigns mu, sigma, and ro values to a variable
    [mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4] = var
    return gamma[0]*ff(x,y,mux1,muy1,sigmax1,sigmay1,rho1,x0,y0,t,sigmax0,sigmay0) + gamma[1]*ff(x,y,mux2,muy2,sigmax2,sigmay2,rho2,x0,y0,t,sigmax0,sigmay0) + gamma[2]*ff(x,y,mux3,muy3,sigmax3,sigmay3,rho3,x0,y0,t,sigmax0,sigmay0) + gamma[3]*ff(x,y,mux4,muy4,sigmax4,sigmay4,rho4,x0,y0,t,sigmax0,sigmay0)


#This section describes the likelihood function, the last component in the Bayesian analytical model
#This assumes an exponential distribution of oil concentration sampling variability around the mean
def LV(x,y,con,gamma,var,x0,y0,t,sigmax0,sigmay0): #defines the likelihood function
    [mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4] = var
    l = 1.0 #first number given to create a loop that can be multiplied together
    for i in range(len(x)):
        s = []
        # for ti in Ti[Ti<t]: # t campaign time; ti spill time 
        la = gamma[0]*ff(x[i],y[i],mux1,muy1,sigmax1,sigmay1,rho1,x0,y0,t,sigmax0,sigmay0) + gamma[1]*ff(x[i],y[i],mux2,muy2,sigmax2,sigmay2,rho2,x0,y0,t,sigmax0,sigmay0) + gamma[2]*ff(x[i],y[i],mux3,muy3,sigmax3,sigmay3,rho3,x0,y0,t,sigmax0,sigmay0) + gamma[3]*ff(x[i],y[i],mux4,muy4,sigmax4,sigmay4,rho4,x0,y0,t,sigmax0,sigmay0)
        #mean of the exponential distribution^, x[i] and y[i] are the field data location points
        s.append(la) #updates the array in s and adds new la to the list
        lam = np.sum(s) #adds all values of s together
        la = 0.0
        if lam > 1e-300:
            la = 1.0/(lam) * np.exp(-1.0/(lam)*con[i]) #likelihood function equation, con[i] is the concentration found at the field data points 
        if abs(la-0.0) > 1e-300: #to ensure that la is not equal to zero
            # print lam
        # l=l*np.exp(-np.log(lam) -1/lam*con[i])
            l = l * la
    return l #returns the likelihood function until the loop finishes


#This section multiplies the likelihood function with the first function to complete the Bayesian analytical model to get a probability
def integ(x,y,x0new,y0new,t,xx,yy,con,gamma,x0,y0,tt,sigmax0,sigmay0,var): #multiplies all summed values together to get a probability 
    #return forwd(x,y,gamma,var)*LV(xx,yy,con,gamma,var)
    return forwd(x,y,gamma,var,x0new,y0new,t,sigmax0,sigmay0)*LV(xx,yy,con,gamma,var,x0,y0,tt,sigmax0,sigmay0)
    # return LV(xx,yy,con,gamma,var,x0,y0,tt,sigmax0,sigmay0,Ti)
    #return forwd(x,y,gamma,var,x0,y0,t,sigmax0,sigmay0)


#This section is created for the Monte Carlo Integration. 
#It creates random variables for the default parameter ranges of the model.
#The use of random variables eliminates the need for the deltas in the original Bayesian model.
def sampler(varinterval): #assigns random values to the parameters to give the new parameters in ff, the ranges were given by Angelica
    while True:
        mux = random.uniform(varinterval[0][0],varinterval[0][1])
        muy = random.uniform(varinterval[1][0],varinterval[1][1])
        sigmax = random.uniform(varinterval[2][0],varinterval[2][1])
        sigmay = random.uniform(varinterval[3][0],varinterval[3][1])
        rho = random.uniform(varinterval[4][0],varinterval[4][1])
        yield (mux,muy,sigmax,sigmay,rho,mux,muy,sigmax,sigmay,rho,mux,muy,sigmax,sigmay,rho,mux,muy,sigmax,sigmay,rho)


#This section produces the combination of gamma's for the 4 patches that will result in the maximum likelihood function.
#It updates the lv list until there are no more possible combinations of gamma. 
#gamma is defined as the mass fraction of total oil in the patches. The sum of gamma should be equal to 1. 
#The possible combinations is ultimately 5^4
def FindBestGamma(u,x,y,con,x0,y0,t,sigmax0,sigmay0,varinterval,Ti): #finding the best possible combination of gammas
    g = np.linspace(0.,1.,u) #divides g equally to get [0,0.25,0.5,0.75,1.0]
    gammaPossible = np.array([seq for seq in itertools.product(g, repeat=u-1) if abs(sum(seq) - 1.0) < 1.0e-4]) #gives different combinations of possible gamma values like [0,0.25,0.25,0.50]
    i = 0
    lv = [] #values are entered into lv as an array and will stop once i > len(gammaPossible)
    while i < len(gammaPossible):
        L = []
        j = 0
        while j < 1000:
            var = np.array(sampler(varinterval).next()) #gives the array of random mu, sigma, and ro values
            tmp = LV(x,y,con,gammaPossible[i],var,x0,y0,t,sigmax0,sigmay0,Ti)
            L.append(tmp) #updates the existing tmp list
            j = j+1 
        lv.append(max(L)) #updates L and is returned to the first []
        i = i+1 #gives new value to initial i and continues until loop stops
    # return np.argmax(np.array(lv))#
    return gammaPossible[np.argmax(np.array(lv))] #returns the gammaPossible that gives the maximum probability 

# def retardationDueOilType(oilType):
#     if oilType == 1.0:
#         retardation = 7.0
#     if oilType == 2.0:
#         retardation = 5.6
#     if oilType == 3.0:
#         retardation = 4.2
#     if oilType == 4.0:
#         retardation = 2.8
#     if oilType == 5.0:
#         retardation = 1.4
#     if oilType == 6.0:
#         retardation = 0.0
#     return retardation


#This section is used to import data that is given from the User in the GUI. 
#It returns the values entered into different arrays to be used for analysis. 
#sigmax0 and sigmay0 were assumed from Angelica's code.
class soscore(Preliminars): #SOSimCore, this imports the data from the user in the GUI
    def __init__(self,InputFileName):
        Preliminars.__init__(self)
        #Input data
        datalist = pd.read_csv(InputFileName) #importing of the excel file
        lat = datalist['lat'] #puts latitude into a list
        lon = datalist['lon'] #puts longitude into a list
        SpillT = datalist['SpillTime'] #puts spill time into a list
        # SampleT = datalist['SampleTime']
        PredictT = datalist['PredictTime'] #puts the prediction time(s) into a list
        Scale = datalist['Scale'] #purpose is to define an area for prediction
        Node = datalist['Node'] #purpose of node is to divide into a grid
        OilType = datalist['OilType'] #puts oil type into list
        lat = np.array(lat[~np.isnan(lat)]) #puts data into an array (isnan double checks that the number is not equal to infinity)
        lon = np.array(lon[~np.isnan(lon)]) #puts data into an array
        SpillT = SpillT[~pd.isnull(SpillT)]
        # SampleT = SampleT[~pd.isnull(SampleT)]
        PredictT = PredictT[~pd.isnull(PredictT)] #proves that Not a Number is not equivalent to infinity
        Scale = np.array(Scale[~np.isnan(Scale)])
        Node = np.array(Node[~np.isnan(Node)])
        OilType = int(OilType[~np.isnan(OilType)])
        sigmax0 = 0.050 #assumed sigmax0
        sigmay0 = 0.050 #assumed sigmay0


        #This section is created to define the spill point and location.
        #It converts the coordinates from decimal degrees to utm
        #self. is used to create a new object in the soscore class
        coord0 = utm.from_latlon(lat, lon) #converts coordinates from degrees to utm for code to read
        x0 = coord0[0]/1000.0
        y0 = coord0[1]/1000.0
        self.x0 = x0 #creates new objects in the class (self.)
        self.y0 = y0
        self.SpillT = SpillT
        self.PredictT = PredictT
        self.Scale = Scale
        self.Node = Node
        self.OilType = OilType
        self.sx0 = sigmax0
        self.sy0 = sigmay0

        self.xclicks = 0 #zero because there is no reflection
        self.yclicks = 0

        self.t = [CalTime(SpillT[0],PredictT[vld]) for vld in range(len(PredictT))] #spill duration, must be less than the prediction time
        self.scale = Scale

        self.lat0 = lat
        self.lon0 = lon


    #This section uploads the different campaign data from field sampling. 
    #It puts the different data into arrays and updates concentration values. 
    def UploadCampaign(self,CampaignFileName): #uploading different field data 
        DLx = []
        DLy = []
        DLcon = []
        st = [] # sample time
        for i in range(len(CampaignFileName)):
            campdata= pd.read_csv(CampaignFileName[i]) #importing the excel file(s) for the field data 
            SampleT = campdata['SampleTime']
            SampleT = SampleT[~pd.isnull(SampleT)]
            DLlat = np.array(campdata["lat"]) #creates an array for the sample latitudes
            DLlon = np.array(campdata["Lon"]) #creates an array for the sample longitudes
            DLc = np.array(campdata["Con"]) #creates an array for the sample concentrations 

            camdatalist = np.array([utm.from_latlon(i,j) for i,j in zip(DLlat,DLlon)])
            DLx.append(np.array(map(float,camdatalist[:,0]))/1000)
            DLy.append(np.array(map(float,camdatalist[:,1]))/1000)
            st.append(CalTime(self.SpillT[0],SampleT[0]))
            conValues = []
            for s in DLc:
                if s == 0.0: #if the field data concentration is zero
                    conValue = 0.1/100.0 #if zero, it gives concentration value of 0.001
                else:
                    conValue = s/100.0 #if not zero, it gives the actual concentration 
                conValues.append(conValue) #updates convalues[] into an array
            DLcon.append(np.array(conValues))
        self.DLx = DLx
        self.DLy = DLy
        self.DLcon = DLcon
        self.st = st


    #This section was taken directly from Angelica's code. 
    #This section takes the input oil type and gives a retardation value.
    #This section also picks the maximum concentration from each campaign (if there are more than 1 campaigns)
    #hiIndex is used to find the respective location of the concentration and the latest sampling time is also found.
    #The time where the maximum concentration is found is added to the retardation to get a new sampling time.
    def retardationDueOilType(self): #defines the oil retardation at the sampling point
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
        B = [max(self.DLcon[vld]) for vld in xrange(len(self.st))] #different campaign data, it picks the maximum concentration for each campaign
        hiIndex = B.index(max(B)) #location for the maximum concentration 
        latestST = max(self.st) #finds latest sampling time
        self.admST = self.st[hiIndex] + self.retardation #sampling time at maximum concentration plus retardation
        self.t = np.array(self.t) - self.admST #prediction time to the maximum concentration campaign


    #Cannot be performed inside retardationDueOilType because that is done when N or S clicking, when no DLx, DLy, DLcon have been input yet.
    #needs to be called after data has been uploaded and processed.
    #This was taken directly from Angelica's code. 
    def x0y0DueSinkingRetardation(self): #given in Angelica's SOSimOPI module, defines the oil characteristic at the spill location
        B = [max(self.DLcon[vld]) for vld in xrange(len(self.st))]#[max(self.DLcon[vld]) for vld in xrange(len(self.st))]
        C = [self.DLcon[vld].argmax() for vld in xrange(len(self.st))]
        x0new = [(self.DLx[vld][C[vld]]) for vld in xrange(len(self.st))]
        y0new = [(self.DLy[vld][C[vld]]) for vld in xrange(len(self.st))]
        tnews = [self.st[vld]*B[vld] for vld in xrange(len(self.st))]
        x0 = self.x0
        y0 = self.y0
        oilType = self.OilType
        distX = np.array([x0new[vld] - self.x0 for vld in xrange(len(self.st))])
        distY = np.array([y0new[vld] - self.y0 for vld in xrange(len(self.st))])
        if oilType == 1.0:
            sunkx0 = (x0 + (7.0*(distX/8.0)))*B
            sunky0 = (y0 + (7.0*(distY/8.0)))*B
        if oilType == 2.0:
            sunkx0 = (x0 + (5.6*(distX/8.0)))*B
            sunky0 = (y0 + (5.6*(distY/8.0)))*B
        if oilType == 3.0:
            sunkx0 = (x0 + (4.2*(distX/8.0)))*B
            sunky0 = (y0 + (4.2*(distY/8.0)))*B
        if oilType == 4.0:
            sunkx0 = (x0 + (2.8*(distX/8.0)))*B
            sunky0 = (y0 + (2.8*(distY/8.0)))*B
        if oilType == 5.0:
            sunkx0 = (x0 + (1.4*(distX/8.0)))*B
            sunky0 = (y0 + (1.4*(distY/8.0)))*B
        if oilType == 6.0:
            sunkx0 = (x0 + (0.0*(distX/8.0)))*B
            sunky0 = (y0 + (0.0*(distY/8.0)))*B
        self.sunkx0 = sum(sunkx0)/sum(B) #average
        self.sunky0 = sum(sunky0)/sum(B)
def CalTime(a,b): #converts dates from excel into the code 
    start = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    ends = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')

    diff = ends - start
    return diff.total_seconds()/86400.


 #These are the default parameters for the code given in Angelica's paper.
 #The values are put into an array to give the varinterval 
 #The Sampler() class can take random values from this array.       
if __name__ == "__main__": #assumed ranges of values from Angelica's code
    Dxmin = 0.01
    Dymin = 0.01
    Dxmax = 0.89    
    Dymax = 0.89 
    vxmin = -3.0
    vymin = -3.0
    vxmax = 3.0
    vymax = 3.0
    romin = -0.99
    romax = 0.99

    varinterval = [[vxmin,vxmax],[vymin,vymax],[Dxmin,Dxmax],[Dymin,Dymax],[romin,romax]] #puts the values into an array for the random assignment in Sampler()
    #Ti = np.linspace(SpillT[0],SpillT[1],Tn)
    #bestgamma=FindBestGamma(5,DLx,DLy,DLc,x0,y0,ST,sigmax0,sigmay0,varinterval,Ti)
    #gamma = bestgamma

    #print bestgamma

    gamma = [0.0,0.0,0.0,1.0]  
    

    #This section uploads the sample input data and the given campaign data.
    #This section also creates the plot to provide the given output. 
    #This section is temporary until the code can be integrated with the GUI.
    a = soscore("datainput.csv")
    a.UploadCampaign(["data2.csv"])
    a.retardationDueOilType()
    a.x0y0DueSinkingRetardation()
    # Calculate GammaPossible
    aargs = a.doAll(Dxmin, Dymin, Dxmax, Dymax, vxmin, vymin, vxmax, vymax, romin, romax)
    newsze = aargs[1]
    valid = aargs[2]
    vx = aargs[3]
    vy = aargs[4]
    Dx = aargs[5]
    Dy = aargs[6]
    ro = aargs[7]
    g =  aargs[8]
    if len(a.st) == 1:
        maxN = len(a.DLcon[0])
    if len(a.st) > 1:
        maxN = 0
        k = 0
        while k < (len(a.DLcon))-1:
            maxN = max(maxN, max(len(a.DLcon[k]), len(a.DLcon[k+1])))
            k += 1
    print "maxN: %s" % maxN
    argus = aargs
    argus.append(a.xclicks)
    argus.append(a.yclicks)
    a.user_nodes_x = 0
    a.user_nodes_y = 0


    #provides the grid for the output image
    x_min = a.lat0 - a.scale[0]
    x_max = a.lat0 + a.scale[0]
    y_min = a.lon0 - a.scale[1]
    y_max = a.lon0 + a.scale[1]
    a.newsze = newsze
    print a.x0,a.y0
    print "Model Size" + str(x_min) +str(x_max)+str(y_min)+str(y_max)
    print a.t, a.admST
    leftc = utm.from_latlon(x_min,y_min)
    rightc = utm.from_latlon(x_max,y_max)
    a.x_min = leftc[0]/1000
    a.x_max = rightc[0]/1000
    a.y_min = leftc[1]/1000
    a.y_max = rightc[1]/1000

    B = [max(a.DLcon[vld]) for vld in xrange(len(a.st))]
    C = [a.DLcon[vld].argmax() for vld in xrange(len(a.st))]
    x0news = [(a.DLx[vld][C[vld]])*B[vld] for vld in xrange(len(a.st))]
    y0news = [(a.DLy[vld][C[vld]])*B[vld] for vld in xrange(len(a.st))]
    x0new = sum(x0news)/sum(B)
    y0new = sum(y0news)/sum(B)
    a.x0new = x0new
    a.y0new = y0new
    print "x0new:", x0new
    print "y0new:", y0new
    
    #divides the scale into equal sections for the output image
    X = np.linspace(a.lat0-a.scale[0],a.lat0+a.scale[0],a.Node+1)
    Y = np.linspace(a.lon0-a.scale[1],a.lon0+a.scale[1],a.Node+1)
    coord = np.array([utm.from_latlon(i,j) for (i,j) in zip(X,Y)])
    x = np.array(map(float,coord[:,0]))/1000
    y = np.array(map(float,coord[:,1]))/1000
    [x,y] = np.meshgrid(x,y)
    [Xp, Yp] = np.meshgrid(X,Y)
    x = np.concatenate(x) #joins a sequence of arrays along an existing axis
    y = np.concatenate(y)
    print a.t
    print a.admST
    print a.DLx[0]
    res=[]
    resfinal = []
    print len(x),len(y)
    for r in range(len(a.t)):
        resa=[]
        for i,j in zip(x,y):
                # def integ(x,y,x0new,y0new,a.t,xx,yy,con,gamma,x0,y0,tt,sigmax0,sigmay0,var):
                result, error = mcint.integrate(lambda v: integ(i,j,a.x0new,a.y0new,a.t[r],a.DLx[0],a.DLy[0],a.DLcon[0],gamma,a.x0,a.y0,a.admST,a.sx0,a.sy0,v), sampler(varinterval), measure=1)
                resa.append(result)
        res.append(resa)
    s=np.array(res)
    for i in range(len(a.t)):
        plt.figure(i)
        l = int(np.sqrt(len(x)))
        plt.contourf(Y,X,s.reshape(l,l),500)
        plt.show()
        # names=r'g%s' %ga
        # G = open(names,"w")
        # np.savetxt(names,resfinal)
        # G.close()

    # print res
    # # print result






    # # Tn=1
    # # Ti = np.linspace(SpillT[0],SpillT[1],1)
    # Ti= [0.]
    # # Ti = np.linspace(SpillT[0],SpillT[1],Tn)
    # #nmc = 10
    # domainsize = 1.0
    # resfinal = []
    # res = [] 
    # for t in PT: #prediction time#
    #     res = []
    #     resa = []
    #     for i in xc:
    #         for j in yc:
    #             result, error = mcint.integrate(lambda v: integ(i,j,DLx,DLy,DLc,gamma,x0,y0,t, ST,sigmax0,sigmay0,v,Ti),sampler(varinterval), measure=domainsize)
    #             resa.append(result)
    #     res.append(resa)
    # resfinal.append(np.sum(res,0))

    # for i in range(len(PT)):
    #     plt.figure(i)
    #     s=resfinal 
    #     s=np.array(s)
    #     # s= s/np.max(s)
    #     # print s
    #     plt.pcolor(mm[:,1],mm[:,0],(s.reshape(len(xc),len(xc))))
    #     plt.colorbar()
    # names=r'g%s' %ga
    # G = open(names,"w")
    # np.savetxt(names,resfinal)
    # G.close()
    # #G.write(str(resfinal))
    # # plt.show() 


   

# print res
# Find the best gamma
# print x0,DLx[85:86]
# print DLc[85:86]
# v = sampler(varinterval)
#vv=v.next()
# print vv
# # print ff(2.0,2.0,0.0,0.0,0.5,0.5,0.5,0,0,0,0.1,0.1)
# print x0\
# print ff(x0+0.01,y0+0.01,vv[0],vv[1],vv[2],vv[3],vv[4],x0,y0,0.0,sigmax0,sigmay0)
# print LV(DLx[40:80],DLy[40:80],DLc[40:80],[0.25,0.25,0.25,0.25],v.next(),x0,y0,7.0,sigmax0,sigmay0)
#bestgamma=FindBestGamma(5,DLx[85:90],DLy[85:90],DLc[85:90],x0,y0,ST,sigmax0,sigmay0,varinterval)
#print bestgamma
# print bestgamma
# gamma = [1.0,0.0,0.0,0.0]
# nmc = 1000
# domainsize = 1.0
# for t in PT:
#     for i in x[1:2]:
#         for j in y[1:2]:
#             result, error = mcint.integrate(lambda v: integ(i,j,DLx,DLy,DLc,gamma,x0,y0,t,ST,sigmax0,sigmay0,v), sampler(), measure=domainsize, n=nmc)
# print result


## THE MAIN PART
## First one need the simulation area， defined as lat，lon min&max
## INPUT DECLARATION
# First we assume all the values are in SI unit
# spillcood = [10,10]
# spillT = 0
# predictT = 10
# modelArea = [[0,0],[20,20]]
# gridN = [25,25]
# campdata= pd.read_csv("data.csv")
# xx = np.array(campdata["lat"])
# yy = np.array(campdata["lon"])
# cc = np.array(campdata["con"])
# tt = np.array(campdata["time"])
# print xx
# print yy
# print cc

# gridx = 10
# gridy = 10
# xc = np.linspace(modelArea[0][0],modelArea[1][0],gridx+1)
# yc = np.linspace(modelArea[0][1],modelArea[1][1],gridy+1)
# xper = np.array(list(itertools.product(xc,yc)))
# x = xper[:,0]
# y = xper[:,1]
# domainsize = 1.0
# t = 0
# nmc = 1000
# #while t < predictT:
# result, error = mcint.integrate(lambda v: integ(x[0],y[0],xx,yy,cc,gamma,v), sampler(), measure=domainsize, n=nmc)





# x=2.0
# y=2.0
# xx=[2.0,2.0]
# yy=[2.0,2.0]
# con=[1.0,0.5]
# gamma = [0.25,0.25,0.25,0.25]
# domainsize = 1.0
# for nmc in [1000,10000]:
#    random.seed(1)
#    result, error = mcint.integrate(lambda v: integ(x,y,xx,yy,con,gamma,v), sampler(), measure=domainsize, n=nmc)
# print result
# bestgamma=FindBestGamma(5,xx,yy,con)
# print bestgamma

