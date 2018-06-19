 # -*- coding: utf-8 -*-

#"A predictive Bayesian data-derived multi-modal Gaussian model of sunken oil mass" Article by Angelica Echavarria-Gregory.
#Angelica's dissertation "Predictive Data-Derived Bayesian Statistic-Transport Model and Simulator of Sunken Oil Mass" can also be used as a reference.
#Reading the article and dissertation can provide you with an understanding of the background statistics and analysis.

from __future__ import division
import sys
sys.path.append('/Users/maryjacketti/Desktop/SOSim/SOSim')


import itertools #The itertools module includes a set of functions for working with iterable (sequence-like) data sets
from math import * #This module provides access to the mathematical functions defined by the C standard.
import numpy as np #This module allows the use of arrays and lists for calculations. 
import random #This module generates a random number in between the given range.
import mcint #This module allows for the Monte Carlo integration 
import utm #This module converts coordinates from degrees to utm and back
import pandas as pd #Allows for the uploading of different excel file campaigns 
import matplotlib.pyplot as plt #Useed for plotting the result 
from multiprocessing import pool #Pool allows for the multiprocessing of the Monte Carlo Integration
import datetime #This module supplies classes for manipulating dates and times in both simple and complex ways
import multiprocessing as mp #Allows multiprocessing
from multiprocessing import cpu_count #Can count how much CPU the computer uses
from multiprocessing.pool import ApplyResult #Applies the results from each multiprocess 
from functools import partial #Used for higher order functions
from sklearn.cluster import KMeans #Clusters the campaigns into groups to form the amount of patches
import multiprocessing 

#This section describes the conditional bivariate Gaussian distribution.
#This is the conditional sampling distribution for the 2-D Bayesian model.
#mux and muy are the 2-D means, sigmax and sigmay are the 2-D covariance matrixes
#DLx and DLy are the sampling locations
#rho is the correlation coefficient
#ff is used to define mux,muy,sigmax,sigmay and returns the value for the conditional bivariate Gaussian distribution
def B_sampling(x,y,mux,muy,sigmax,sigmay,rho): #definition of Bm in the conditional probability equation
    Bs=((((x-mux))**(2.0))/((sigmax)**2.0))+((((y-muy))**(2.0))/((sigmay)**2.0))-((2.0*(rho)*(x-mux)*(y-muy))/(sigmax*sigmay))
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

def forwd(x,y,gamma,var,x0,y0,t,sigmax0,sigmay0): #assigns mu, sigma, and ro values to a variable
    [mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4] = var
    return gamma[0]*ff(x,y,mux1,muy1,sigmax1,sigmay1,rho1,x0[0],y0[0],t,sigmax0,sigmay0) + gamma[1]*ff(x,y,mux2,muy2,sigmax2,sigmay2,rho2,x0[1],y0[1],t,sigmax0,sigmay0) + gamma[2]*ff(x,y,mux3,muy3,sigmax3,sigmay3,rho3,x0[2],y0[2],t,sigmax0,sigmay0) + gamma[3]*ff(x,y,mux4,muy4,sigmax4,sigmay4,rho4,x0[3],y0[3],t,sigmax0,sigmay0)


#This section describes the likelihood function, the last component in the Bayesian analytical model
#This assumes an exponential distribution of oil concentration sampling variability around the mean
def LV(DLx,DLy,DLcon,gamma,var,x0,y0,t,sigmax0,sigmay0): #defines the likelihood function
    [mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4] = var
    l = 1.0 #first number given to create a loop that can be multiplied together
    stn = 0
    while stn < len(t):
        for i in range(len(DLx)):
            s = []
            # for ti in Ti[Ti<t]: # t campaign time; ti spill time 
            la = gamma[0]*ff(DLx[stn][i],DLy[stn][i],mux1,muy1,sigmax1,sigmay1,rho1,x0,y0,t[stn],sigmax0,sigmay0) + gamma[1]*ff(DLx[stn][i],DLy[stn][i],mux2,muy2,sigmax2,sigmay2,rho2,x0,y0,t[stn],sigmax0,sigmay0) + gamma[2]*ff(DLx[stn][i],DLy[stn][i],mux3,muy3,sigmax3,sigmay3,rho3,x0,y0,t[stn],sigmax0,sigmay0) + gamma[3]*ff(DLx[stn][i],DLy[stn][i],mux4,muy4,sigmax4,sigmay4,rho4,x0,y0,t[stn],sigmax0,sigmay0)
            #mean of the exponential distribution^, x[i] and y[i] are the field data location points
            s.append(la) #updates the array in s and adds new la to the list
            lam = np.sum(s) #adds all values of s together
            la = 0.0
            if lam > 1e-300:
                la = 1.0/(lam) * np.exp(-1.0/(lam)*DLcon[i]) #likelihood function equation, con[i] is the concentration found at the field data points 
            if abs(la-0.0) > 1e-300: #to ensure that la is not equal to zero
            # print lam
        # l=l*np.exp(-np.log(lam) -1/lam*con[i])
                l = l * la
        stn = stn + 1
    return l #returns the likelihood function until the loop finished


#This section multiplies the likelihood function with the first function to complete the Bayesian analytical model to get a probability
def integ(x,y,x0new,y0new,t,DLx,DLy,DLcon,gamma,tt,x0,y0,sigmax0,sigmay0,var): #multiplies all summed values together to get a probability 
    return forwd(x,y,gamma,var,x0new,y0new,t,sigmax0,sigmay0)* LV(DLx,DLy,DLcon,gamma,var,x0,y0,tt,sigmax0,sigmay0)
    # return LV(xx,yy,con,gamma,var,x0,y0,tt,sigmax0,sigmay0,Ti)
    #return forwd(x,y,gamma,var,x0,y0,t,sigmax0,sigmay0)

#The integ2 is used for when a prediction is wanted before any sample data has been collected.
def integ2(x,y,x0new,y0new,t,gamma,sigmax0,sigmay0,var):
    #return forwd(x,y,gamma,var)*LV(xx,yy,con,gamma,var)
    return forwd(x,y,gamma,var,x0new,y0new,t,sigmax0,sigmay0)


#This section is created for the Monte Carlo Integration. 
#It creates random variables for the default parameter ranges of the model.
#The use of random variables eliminates the need for the deltas in the original Bayesian model.
def sampler(varinterval): #assigns random values to the parameters to give the new parameters in ff, the ranges were given by Angelica
    while True:
        mux1 = random.uniform(varinterval[0][0],varinterval[0][1])
        mux2 = random.uniform(varinterval[0][0],varinterval[0][1])
        mux3 = random.uniform(varinterval[0][0],varinterval[0][1])
        mux4 = random.uniform(varinterval[0][0],varinterval[0][1])
        muy1 = random.uniform(varinterval[1][0],varinterval[1][1])
        muy2 = random.uniform(varinterval[1][0],varinterval[1][1])
        muy3 = random.uniform(varinterval[1][0],varinterval[1][1])
        muy4 = random.uniform(varinterval[1][0],varinterval[1][1])
        sigmax1 = random.uniform(varinterval[2][0],varinterval[2][1])
        sigmax2 = random.uniform(varinterval[2][0],varinterval[2][1])
        sigmax3 = random.uniform(varinterval[2][0],varinterval[2][1])
        sigmax4 = random.uniform(varinterval[2][0],varinterval[2][1])
        sigmay1 = random.uniform(varinterval[3][0],varinterval[3][1])
        sigmay2 = random.uniform(varinterval[3][0],varinterval[3][1])
        sigmay3 = random.uniform(varinterval[3][0],varinterval[3][1])
        sigmay4 = random.uniform(varinterval[3][0],varinterval[3][1])
        rho1 = random.uniform(varinterval[4][0],varinterval[4][1])
        rho2 = random.uniform(varinterval[4][0],varinterval[4][1])
        rho3 = random.uniform(varinterval[4][0],varinterval[4][1])
        rho4 = random.uniform(varinterval[4][0],varinterval[4][1])
        yield (mux1,muy1,sigmax1,sigmay1,rho1,mux2,muy2,sigmax2,sigmay2,rho2,mux3,muy3,sigmax3,sigmay3,rho3,mux4,muy4,sigmax4,sigmay4,rho4)


#This section produces the combination of gamma's for the 4 patches that will result in the maximum likelihood function.
#It updates the lv list until there are no more possible combinations of gamma. 
#gamma is defined as the mass fraction of total oil in the patches. The sum of gamma should be equal to 1. 
#The possible combinations is ultimately 5^4
#The gammavalid is related to the x0new since the amount of patches has been determined. 
def FindBestGamma(u,DLx,DLy,DLcon,x0,y0,t,sigmax0,sigmay0,varinterval): #finding the best possible combination of gammas
    g = np.linspace(0.,1.,u) #divides g equally to get [0,0.25,0.5,0.75,1.0]
    gammaPossible = np.array([seq for seq in itertools.product(g, repeat=u-1) if abs(sum(seq) - 1.0) < 1.0e-4]) #gives different combinations of possible gamma values like [0,0.25,0.25,0.50]
    m=np.ones(patch)
    print m
    gammavalid = [i for i in gammaPossible if sum(i[0:patch] > 0.) == patch and sum(i > 0.) == patch]
    print gammavalid
    i = 0
    lv = [] #values are entered into lv as an array and will stop once i > len(gammaPossible)
    while i < len(gammavalid):
        L = []
        j = 0
        while j < 1000:
            var = np.array(sampler(varinterval).next()) #gives the array of random mu, sigma, and ro values
            tmp = LV(DLx,DLy,DLcon,gammavalid[i],var,x0,y0,t,sigmax0,sigmay0)
            L.append(tmp) #updates the existing tmp list
            j = j+1 
        lv.append(max(L)) #updates L and is returned to the first []
        i = i+1 #gives new value to initial i and continues until loop stops
    # return np.argmax(np.array(lv))#
    return gammavalid[np.argmax(np.array(lv))]

class Preliminars: # SOSim
    def __init__(self): 
        self.w = 4
        self.u = self.w + 1
        self.delta = 0
        self.args = []
        self.discarded = 0.0
        self.valid = 0.0
        self.GammaPossible = []


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
        B = max(max(self.DLcon)) #different campaign data, it picks the maximum concentration for each campaign
        self.hiIndex = [i for i, j in enumerate(max(self.DLcon)) if j == B] #location for the maximum concentration 
        latestST = max(self.st) #finds latest sampling time
        self.admST = latestST + self.retardation #sampling time at maximum concentration plus retardation
        self.t = np.array(self.t) - self.admST #prediction time to the maximum concentration campaign


    #Cannot be performed inside retardationDueOilType because that is done when N or S clicking, when no DLx, DLy, DLcon have been input yet.
    #needs to be called after data has been uploaded and processed.
    #This was taken directly from Angelica's code. 
    def x0y0DueSinkingRetardation(self): #given in Angelica's SOSimOPI module, defines the oil characteristic at the spill location
        B = [max(self.DLcon[vld]) for vld in xrange(len(self.st))]
        hiIndex = B.index(max(B))
        self.DLcon[hiIndex]        
        C = [i for i, j in enumerate(self.DLcon[hiIndex].tolist()) if j == max(B)]
        x0news = self.DLx[hiIndex][C]
        y0news = self.DLy[hiIndex][C]
        DL = np.array([[i,j] for i, j in zip(x0news,y0news)])
        if len(DL) < 4.:
            n_clusters = len(DL)
        else:
            n_clusters = 4
        estimator = KMeans(n_clusters)#构造聚类器
        estimator.fit(DL)#聚类
        label_pred = estimator.labels_ #获取聚类标签
        centroids = estimator.cluster_centers_ #获取聚类中心
        inertia = estimator.inertia_ # 获取聚类准则的总和
        
        x0new = centroids[:,0]
        y0new = centroids[:,1]
        self.x0new = np.pad(x0new,(0,4-len(DL)),'constant')
        self.y0new = np.pad(y0new,(0,4-len(DL)),'constant')
        self.patch = n_clusters

        plt.plot(x0news,y0news,'.')
        plt.plot(centroids[:,0],centroids[:,1],'*')

        x0 = self.x0
        y0 = self.y0
        oilType = self.OilType
        #distX = np.array([x0new[vld] - self.x0 for vld in xrange(len(self.st))])
        #distY = np.array([y0new[vld] - self.y0 for vld in xrange(len(self.st))])
        distX = np.array(x0new - x0)
        distY = np.array(y0new - y0)
        if oilType == 1.0:
            sunkx0 = (x0 + (7.0*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (7.0*(np.array(distY)/8.0)))*B
        if oilType == 2.0:
            sunkx0 = (x0 + (5.6*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (5.6*(np.array(distY)/8.0)))*B
        if oilType == 3.0:
            sunkx0 = (x0 + (4.2*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (4.2*(np.array(distY)/8.0)))*B
        if oilType == 4.0:
            sunkx0 = (x0 + (2.8*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (2.8*(np.array(distY)/8.0)))*B
        if oilType == 5.0:
            sunkx0 = (x0 + (1.4*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (1.4*(np.array(distY)/8.0)))*B
        if oilType == 6.0:
            sunkx0 = (x0 + (0.0*(np.array(distX)/8.0)))*B
            sunky0 = (y0 + (0.0*(np.array(distY)/8.0)))*B
        self.sunkx0 = sum(sunkx0)/len(x0new) #average
        self.sunky0 = sum(sunky0)/len(y0new)

    #This section is called when there has been no sample data and the likelihood function cannot be determined. 
    def NO_campaign(self):
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
        self.st = 0
        self.admST = self.st + self.retardation
        self.t = np.array(self.t) - self.admST # time to the maximum concentration campaign
        self.x0new = np.ones(4)*self.x0
        self.y0new = np.ones(4)*self.y0

def CalTime(a,b): #converts dates from excel into the code 
    start = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    ends = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')

    diff = ends - start
    return diff.total_seconds()/86400.

#This section allows for the use of multiprocessing.
#When used, it decreases the run time from about 5 hours for an integration calculation of 100,000 times to 1 hour. 
def job(a,parameter):
    x,y = parameter
    result1, error = mcint.integrate(lambda v: integ(x,y,a.x0new,a.y0new,a.t[0],a.DLx,a.DLy,a.DLcon,a.gamma,a.sunkx0,a.sunky0,a.st,a.sx0,a.sy0,v), sampler(a.varinterval), measure=1,n=1000)
    return result1

def multicore(a,parameter):
        pool = mp.Pool(4)
        #res = pool.map(job())
        # res = pool.apply_async(job())
        # multi_res = [pool.apply_async(job(),(parameter[i]))for i in range(10)]
        # mean([res.get() for res in multi_res])
        # pa = [[0,1],[1,2]]
        res = pool.map(partial(job,a),parameter)
        return res


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
    romin = -0.999
    romax = 0.999

    varinterval = [[vxmin,vxmax],[vymin,vymax],[Dxmin,Dxmax],[Dymin,Dymax],[romin,romax]] #puts the values into an array for the random assignment in Sampler()
    #Ti = np.linspace(SpillT[0],SpillT[1],Tn)
  
  
    

    #This section uploads the sample input data and the given campaign data.
    #This section also creates the plot to provide the given output. 
    #This section is temporary until the code can be integrated with the GUI.
    a = soscore("datainput.csv")
    a.UploadCampaign(["DBL input1.csv"])
    a.retardationDueOilType()
    a.x0y0DueSinkingRetardation()
    #a.NO_campaign()

    print a.st
    # Calculate GammaPossible
    #aargs = a.doAll(Dxmin, Dymin, Dxmax, Dymax, vxmin, vymin, vxmax, vymax, romin, romax)
    #newsze = aargs[1]
    #valid = aargs[2]
    #vx = aargs[3]
    #vy = aargs[4]
    #Dx = aargs[5]
    #Dy = aargs[6]
    #ro = aargs[7]
    #g =  aargs[8]
    if len(a.st) == 1:
        maxN = len(a.DLcon[0])
    if len(a.st) > 1:
        maxN = 0
        k = 0
        while k < (len(a.DLcon))-1:
            maxN = max(maxN, max(len(a.DLcon[k]), len(a.DLcon[k+1])))
            k += 1
    print "maxN: %s" % maxN
    #argus = aargs
    #argus.append(a.xclicks)
    #argus.append(a.yclicks)
    #a.user_nodes_x = 0
    #a.user_nodes_y = 0

    gamma=FindBestGamma(a.u,a.DLx,a.DLy,a.DLcon,a.sunkx0,a.sunky0,a.st,a.sx0,a.sy0,varinterval,a.patch) 
    a.gamma = gamma
    print gamma

    a.varinterval = varinterval

    #provides the grid for the output image
    x_min = a.lat0 - a.scale[0]
    x_max = a.lat0 + a.scale[0]
    y_min = a.lon0 - a.scale[1]
    y_max = a.lon0 + a.scale[1]
    #a.newsze = newsze
    print a.x0,a.y0
    print "Model Size" + str(x_min) +str(x_max)+str(y_min)+str(y_max)
    print a.t, a.admST
    leftc = utm.from_latlon(x_min,y_min)
    rightc = utm.from_latlon(x_max,y_max)
    a.x_min = leftc[0]/1000
    a.x_max = rightc[0]/1000
    a.y_min = leftc[1]/1000
    a.y_max = rightc[1]/1000

    #B = max(max(a.DLcon)) #[max(self.DLcon[vld]) for vld in xrange(len(self.st))]
    #C = [i for i, j in enumerate(max(a.DLcon)) if j == B]
    #DLx=max(a.DLx)
    #DLy=max(a.DLy)
    #x0new = np.array(DLx[C])
    #y0new = np.array(DLy[C])
    #x0news = [(a.DLx[vld][C[vld]])*B[vld] for vld in xrange(len(a.st))]
    #y0news = [(a.DLy[vld][C[vld]])*B[vld] for vld in xrange(len(a.st))]
    #x0new = sum(x0news)/sum(B)
    #y0new = sum(y0news)/sum(B)
    #a.x0new = x0new
    #a.y0new = y0new
    #print "x0new:", x0new
    #print "y0new:", y0new
    
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
    print a.DLx
    print a.DLy
    print a.DLcon
    print a.x0new
    print a.sunkx0
    
    res=[]
    resfinal = []
    print len(x),len(y)

    for r in range(len(a.t)): 
        resa=[]
        parameter = zip(x,y)
        resa = [multcore(a,parameter) for m in range(5)]
        b = len(resa)
        sum = 0
        for i in resa:
            sum = sum + np.array(i)

        res.append(sum)

        #for i,j in zip(x,y):
                # def integ(x,y,x0new,y0new,a.t,xx,yy,con,gamma,x0,y0,tt,sigmax0,sigmay0,var):
                #result, error = mcint.integrate(lambda v: integ(i,j,a.x0new[0],a.y0new[0],a.t[r],a.DLx[0],a.DLy[0],a.DLcon[0],gamma,a.sunkx0,a.sunky0,a.admST,a.sx0,a.sy0,v), sampler(varinterval), measure=1)
                #result2, error = mcint.integrate(lambda v: integ(i,j,a.x0new[1],a.y0new[1],a.t[r],a.DLx[0],a.DLy[0],a.DLcon[0],gamma,a.sunkx0,a.sunky0,a.admST,a.sx0,a.sy0,v), sampler(varinterval), measure=1)
                #result = result1 + result2
                #resa.append(result)
        #res.append(resa)

    s=np.array(res)
    print np.max(s)
    for i in range(len(a.t)):
        plt.figure(i)
        l = int(np.sqrt(len(x)))
        plt.contourf(Y,X,s.reshape(l,l),500)
        plt.show()
  



