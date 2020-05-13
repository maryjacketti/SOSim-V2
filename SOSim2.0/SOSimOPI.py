"""

Operative and Processing Interface - OPI module of SOSim application.
Copyright (C) 2019 M. Jacketti & C. Ji

This file is part of SOSim - Subsurface Oil Simulator.

    SOSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    SOSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SOSim.  If not, see <http://www.gnu.org/licenses/>.

This code was developed as computational wrapping for a project funded
by a grant from GoMRI.
 
It is also a Deliverable of PhD Theses and dissertations for PhD degrees of Mary Jacketti and Chao Ji.
University of Miami,
May 2018 - June 2022.

For development support contact m.jacketti@miami.edu


"""
PlayPath = "Backstage"
qgis_prefix = "C:/Program Files/QGIS 2.18/apps/qgis-ltr"
SOSimPath = ""
PlayPath = "Backstage"
ResultsPath = "Results"

# myfile = ''
myfile_list = []
myfile1_list = []
myfile2_list = []
global cur
k_zt = 0


#__________________________________________________________________________________________________________________________
# Imports:
import sys
import os
import re
import math
import numpy
import calendar
import string
import time
import shutil
from math import *
from numpy import *
import pickle
from qgis.core import *
from qgis.gui import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import SOSimgui as SOSimgui
from PyQt4.QtCore import QFileInfo,QSettings
from ui_Options import Ui_MyDialog
import SOSimsubmerged as submerged
import SOSimsunken as sunken
from netCDF4 import Dataset
import numpy as np
import cv2 

#___________________________________________________________________________________________________________________________
# Global Functions:

def LatLongConverter(longitude, latitude): 
    """Code to convert WGS coordinates to UTM coordinates"""
    lon = longitude
    lat = latitude
    # Constants:
    # Datum Constants:
    a = 6378137.0               # equatorial radius
    b = 6356752.314             # polar radius	
    f = 0.003352811             # flattening	
    invf = 1.0/f                # inverse flattening
    rm = (a*b)**(1.0/2.0)       # Mean radius
    k0 = 0.9996                 # scale factor	
    ecc = sqrt(1.0-(b/a)**2.0)    # eccentricity	
    ecc2 = ecc*ecc/(1.0-ecc*ecc)
    n = (a-b)/(a+b)
    new = pi/(180.0*3600.0)
    # Meridional Arc Constants		
    A0 = a*(1.0-n+(5.0*n*n/4.0)*(1.0-n) +(81.0*n**4.0/64.0)*(1.0-n))
    B0 = (3.0*a*n/2.0)*(1.0 - n - (7.0*n*n/8.0)*(1.0-n) + 55.0*n**4.0/64.0)
    C0 = (15.0*a*n*n/16.0)*(1.0 - n +(3.0*n*n/4.0)*(1.0-n))
    D0 = (35.0*a*n**3.0/48.0)*(1.0 - n + 11.0*n*n/16.0)	
    E0 = (315.0*a*n**4.0/51.0)*(1.0-n)
    # Calculations:
    if lon > 0.0:
        zone = int(lon/6)+31
    if lon < 0.0:
        zone = int((180+lon)/6)+1
    lonCM = (6*zone)-183
    Dlon = (lon-lonCM)*pi/180.0
    lonrad = lon*pi/180.0
    latrad = lat*pi/180.0
    curvatureR1 = a*(1.0-ecc*ecc)/((1.0-(ecc*sin(latrad))**2.0)**(3.0/2.0))
    curvatureR2 = a/((1.0-(ecc*sin(latrad))**2.0)**(1.0/2.0))
    MeridArc = A0*latrad - B0*sin(2.0*latrad) + C0*sin(4.0*latrad) - D0*sin(6.0*latrad) + E0*sin(8.0*latrad)
    k1 = MeridArc*k0
    k2 = curvatureR2*sin(latrad)*cos(latrad)/2.0
    k3 = ((curvatureR2*sin(latrad)*cos(latrad)**3.0)/24.0)*(5.0-tan(latrad)**2.0+9.0*ecc2*cos(latrad)**2.0+4.0*ecc2**2.0*cos(latrad)**4.0)*k0
    k4 = curvatureR2*cos(latrad)*k0
    k5 = (cos(latrad))**3.0*(curvatureR2/6.0)*(1.0-tan(latrad)**2.0+ecc2*cos(latrad)**2.0)*k0
    k6 = ((Dlon)**6.0*curvatureR2*sin(latrad)*cos(latrad)**5.0/720.0)*(61.0-58.0*tan(latrad)**2.0+tan(latrad)**4.0+270.0*ecc2*cos(latrad)**2.0-330.0*ecc2*sin(latrad)**2.0)*k0
    rawNorth = k1+(k2*Dlon**2.0)+(k3*Dlon**4.0)
    if rawNorth < 0.0:
        North = 10000000.0 + rawNorth
    else:
        North = rawNorth
    East = 500000.0+(k4*Dlon)+(k5*Dlon**3.0)
    location = [East/1000.0, North/1000.0, zone] # in km.
    return location

def UTMConverter(easting, northing, zone):
    """Code to conver UTM coordinates to WGS coordinates"""
    # Constants:
    # Datum Constants:
    a = 6378137.0               # equatorial radius
    b = 6356752.314             # polar radius		
    k0 = 0.9996                 # scale factor	
    ecc = sqrt(1.0-(b/a)**2.0)    # eccentricity	
    ecc2 = ecc*ecc/(1.0-ecc*ecc)
    # For calculations:
    e1 = (1.0-(1.0-ecc*ecc)**(1.0/2.0))/(1.0+(1.0-ecc*ecc)**(1.0/2.0))
    C1 = 3.0*e1/2.0-27.0*e1**3.0/32.0
    C2 = 21.0*e1**2.0/16.0-55.0*e1**4.0/32.0
    C3 = 151.0*e1**3.0/96.0
    C4 = 1097.0*e1**4.0/512.0
    if northing >= 0.0:
        corrNorth = northing
    else:
        corrNorth = 10000000.0 - northing
    eastPrime = 500000.0 - easting
    arcLength = northing/k0
    mu = arcLength/(a*(1.0-ecc**2.0/4.0-3.0*ecc**4.0/64.0-5.0*ecc**6.0/256.0))
    footprintLat = mu+C1*sin(2.0*mu)+C2*sin(4.0*mu)+C3*sin(6.0*mu)+C4*sin(8.0*mu)
    K1 = ecc2*cos(footprintLat)**2.0
    T1 = tan(footprintLat)**2.0
    N1 = a/(1.0-(ecc*sin(footprintLat))**2.0)**(1.0/2.0)
    R1 = a*(1.0-ecc*ecc)/(1.0-(ecc*sin(footprintLat))**2.0)**(3.0/2.0)
    D = eastPrime/(N1*k0)
    # Coeficients for calculating latitude:
    coef1 = N1*tan(footprintLat)/R1
    coef2 = D*D/2.0
    coef3 = (5.0+3.0*T1+10.0*C1-4.0*C1*C1-9.0*ecc2)*D**4.0/24.0
    coef4 = (61.0+90.0*T1+298.0*C1+45.0*T1*T1-252.0*ecc2-3.0*C1*C1)*D**6.0/720.0
    # Coefficients for calculating longitude:
    coef5 = D
    coef6 = (1.0+2.0*T1+C1)*D**3.0/6.0
    coef7 = (5.0-2.0*C1+28.0*T1-3.0*C1**2.0+8.0*ecc2+24.0*T1**2.0)*D**5.0/120.0
    deltalong = (coef5-coef6+coef7)/cos(footprintLat)
    zoneCM = 6.0*zone-183.0
    lat = 180.0*(footprintLat-coef1*(coef2+coef3+coef4))/pi
    if northing >= 0.0:
        lat = lat
    else:
        lat = -lat
    lon = zoneCM - deltalong*180.0/pi
    return [lon, lat]

def CalTime(a,b):
    start = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    ends = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    diff = ends - start
    return diff.total_seconds()/86400.
#__________________________________________________________________________________________________________________________
# QApplication Object:
__version__ = "1.0.0"
#__________________________________________________________________________________________________________________________
# MAC platform accessibility:


#__________________________________________________________________________________________________________________________
class SOSimMainWindow(QMainWindow, SOSimgui.Ui_SOSimMainWindow):
    """This class represents the main window and inherits from both the QMainWindow widget and the QtDesigner file."""
    
    def __init__(self, parent=None): 
        super(SOSimMainWindow, self).__init__(parent)

        global myfile_list
        global myfile1_list
        global myfile2_list
        myfile_list = []
        myfile1_list = []
        myfile2_list = []
        self.ourinformation = {}
        self.ourinformation['CampaignButton'] = []
        self.ourinformation['OurTime'] = []
        self.ourinformation['HydroButton'] = []

        # All in ui_SOMSim.py gets imported and GUI initialized:
        self.setupUi(self)
        self.popDialog = myDialog()

        # Create map canvas:
        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(QColor(0,0,140))
        self.canvas.enableAntiAliasing(True)
        self.canvas.show()
        # Add the canvas to its framed layout created with QtDesigner:
        self.LayoutMap.addWidget(self.canvas)

        # Create global, small map canvas:
        self.globalcanvas = QgsMapCanvas()
        self.globalcanvas.setCanvasColor(QColor(0,0,140))
        self.globalcanvas.enableAntiAliasing(True)
        self.globalcanvas.show()
        # Add the global, small canvas to its framed layout created with QtDesigner:
        self.LayoutGlobal.addWidget(self.globalcanvas)

        # Create canvas for the variable legend:
        self.legendcanvas = QgsMapCanvas()
        self.legendcanvas.setCanvasColor(QColor(250,250,250))
        self.legendcanvas.enableAntiAliasing(True)
        self.legendcanvas.show()
        # Add the legend canvas to its framed layout created with QtDesigner:
        self.LayoutLegend.addWidget(self.legendcanvas)
#______________________new legend______________
        self.outlegendcanvas = QgsMapCanvas()
        self.outlegendcanvas.setCanvasColor(QColor(250,250,250))
        self.outlegendcanvas.enableAntiAliasing(True)
        self.outlegendcanvas.show()        
        self.LayoutLegendHor.addWidget(self.outlegendcanvas)

#______________________________________________
        # create the actions behaviours
        self.connect(self.actionAddLayer, SIGNAL("triggered()"), self.addRasterImage)
        self.connect(self.actionZoomIn, SIGNAL("triggered()"), self.zoomIn)
        self.connect(self.actionZoomOut, SIGNAL("triggered()"), self.zoomOut)
        self.connect(self.actionPan, SIGNAL("triggered()"), self.pan)
        self.connect(self.actionCaptureCoordinates, SIGNAL("triggered()"), self.captureCoords)
        self.connect(self.actionSave_Image, SIGNAL("triggered()"), self.fileSaveAsImage)
        self.connect(self.actionSave_Calibration_As, SIGNAL("triggered()"), self.fileSaveCalibrationAs)
        self.connect(self.actionCurrent_Image, SIGNAL("triggered()"), self.filePrint)
        self.connect(self.actionQuit, SIGNAL("triggered()"), self.fileQuit)
        self.connect(self.actionNew, SIGNAL("triggered()"), self.fileNew)
        self.connect(self.actionExisting_Output_Image, SIGNAL("triggered()"), self.addRasterImage)
        self.connect(self.actionOpen, SIGNAL("triggered()"), self.addRasterImage)
        self.connect(self.actionSave, SIGNAL("triggered()"), self.fileSave)
        self.connect(self.actionDefaultSettings, SIGNAL("triggered()"), self.optionsDefSettings)
        

        # create file toolbar:
        self.fileToolbar = self.addToolBar("File");
        self.fileToolbar.setObjectName("FileToolBar")
        self.fileToolbar.addAction(self.actionOpen)
        self.fileToolbar.addAction(self.actionSave)
        self.fileToolbar.addAction(self.actionNew)
        
        # create map toolbar and place it to the right of the canvas:
        self.mapToolbar = self.addToolBar("Map") #changed by the following line to put it vertical
        self.mapToolbar.setObjectName("MapToolBar")
        self.mapToolbar.addAction(self.actionAddLayer)
        self.mapToolbar.addAction(self.actionCaptureCoordinates)
        self.mapToolbar.addAction(self.actionPan)
        self.mapToolbar.addAction(self.actionZoomIn)
        self.mapToolbar.addAction(self.actionZoomOut)


        # Create the map tools
        self.toolPan = QgsMapToolPan(self.canvas)
        self.toolPan.setAction(self.actionPan)
        self.toolZoomIn = QgsMapToolZoom(self.canvas, False) # false = in
        self.toolZoomIn.setAction(self.actionZoomIn)
        self.toolZoomOut = QgsMapToolZoom(self.canvas, True) # true = out
        self.toolZoomOut.setAction(self.actionZoomOut)
        self.toolCaptureCoordinates = QgsMapToolEmitPoint(self.canvas)
        self.toolCaptureCoordinates.setAction(self.actionCaptureCoordinates)
        
        
        #Scale options
        self.ScaleFrame.hide()
        self.SureButton.hide()
        # Nodes options:
        self.NodesFrame.hide()
        self.connect(self.UserDefinedNodesRadioButton, SIGNAL("toggled(bool)"), self.NodesFrame, SLOT("setVisible(bool)"))
       
        # Layerset:
        self.layers = []
        
        # Show the world base map and base legend:
        self.MyWorldLayer(ext = True)
        self.MyWorldLayerGlobalCanvas()
        self.MyLegendLoad("Data/LegendLandOcean.jpg", "LegendLandOcean.jpg")

        # Other missing in ui:
        self.lineEdit.setAlignment(Qt.AlignHCenter)
        self.lineEdit.setText(str(0.0)+ " ,  " +str(0.0))

        self.RecalcButton.setEnabled(True)

        self.NodataButton.setVisible(False)
        self.UTMButton.setVisible(False)
        self.DecimalButton.setVisible(False)


#___________________________________________________________________________________________________________________
        # FOR THE CORE: passing variables:

        self.lon0 = 0.0
        self.lat0 = 0.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.zone0 = 1
        
        self.DLx = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.DLy = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.DLcon = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.DLzone = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
        self.st = []
        self.allCampaignIndices = []
        self.importedCalibration = bool
        
        self.sx0 = 0.050 # in km, = to 50 meters.
        self.sy0 = 0.050

        self.TimeSamp = zeros(4)
        self.t = []
        self.filename = None
        self.admST = 0
        self.retardation = 0.5
        self.sunkx0 = 0.0
        self.sunky0 = 0.0

        self.x_max = 0.0
        self.x_min = 0.0
        self.y_max = 0.0
        self.y_min = 0.0

        self.markers = [] 
        self.markersglobal = []
        self.northmarker = []
        self.xclicks = []
        self.yclicks = []
        self.openOcean = False

        #self.morning = [self.t[0], 0] #### REVISE ACCORDING TO MIN(ST)
        self.morning = [0, 0]
        self.genForward = self.nextTimeGenerator()
        #self.evening = [self.t[len(self.t)-1], len(self.t)]
        self.evening = [0, 0]
        self.genBackward = self.prevTimeGenerator()

        # FOr passing from Run method to Recalculate method:
        self.LikelihoodFunction = []
        self.args = []

        # To self.RecalcButton.setEnabled(True) and to keep print settings:
        self.printer = None

        #print len(self.layers)




#_____________________________________________________________________________________________________________________
    # Methods to print, save image, etc.


    def optionsDefSettings(self):
        self.popDialog.show()


    def filePrint(self):
        if self.printer is None:
            self.printer = QPrinter(QPrinter.HighResolution)
            self.printer.setPageSize(QPrinter.Letter)
        form = QPrintDialog(self.printer, self)
        if form.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.canvas.size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.drawImage(0, 0, self.canvas)

    def fileQuit(self):
        QApplication.closeAllWindows()


    def fileNew(self):
        SOSimMainWindow().show()


    def fileSave(self):
        what = QMessageBox.question(self, "SOSim - Save %s" % self.spillName(),
                                     "You are about to save your project's current calibration file. Press 'No' if you whish to save your output map instead.",
                                     QMessageBox.Yes|QMessageBox.No)
        what
        if what == QMessageBox.Yes:
            self.fileSaveCalibrationAs()
        if what == QMessageBox.No:
            self.fileSaveAsImage()


    def fileSaveCalibrationAs(self):
        self.filename = "YourCalibration %s.txt" % self.spillName()
        fname = self.filename if self.filename is not None else "."
        format = "*.txt"
        fname = unicode(QFileDialog.getSaveFileName(self,
                                                    "SOSim - Save Calibration As...",
                                                    fname,
                                                    "Calibration files (%s)" % " ".join(format)))
        if fname:
            if "." not in fname:
                fname += ".txt"
            shutil.copy("my_LF.txt", fname)
            shutil.copy("my_GammaCombIndex.txt", fname.replace(".txt", "") + "GammaCombIndex.txt")
            shutil.copy("my_MaxST.txt", fname.replace(".txt", "") + "MaxST.txt")
            shutil.copy("my_DLx.txt", fname.replace(".txt", "") + "DLx.txt")
            shutil.copy("my_DLy.txt", fname.replace(".txt", "") + "DLy.txt")
            shutil.copy("my_DLcon.txt", fname.replace(".txt", "") + "DLcon.txt")
            
            
    def fileSaveAsImage(self):
        self.filename = "Spill %s .png" % self.spillName()
        fname = self.filename if self.filename is not None else "."
        formats = ["*.%s" % unicode(format).lower() for format in QImageWriter.supportedImageFormats()]
        fname = unicode(QFileDialog.getSaveFileName(self,
                                                    "SOSim - Save Image As",
                                                    fname,
                                                    "All Image Files (%s)" % " ".join(formats)))
        if fname:
            if "." not in fname:
                fname += ".png"
            self.filename = fname
            self.canvas.saveAsImage(self.filename)
            
#__________________________________________________________________________________________________________________________
    # Methods for the Spill Name and Oil Type
    
    def spillName(self):
        a =['/', ':', '*', '?', '"','<','>','|']
        for i in a:
            if i in unicode(self.SpillNameEdit.text()):
                QMessageBox.information(self, "SOSim - Invalid Character", "This name cannot contain any of the following characters: '/','\',':','*','?','<','>','|', please change name accordingly.")
                self.SpillNameEdit.selectAll()
                self.SpillNameEdit.setFocus()
            else:
                spillname = unicode(self.SpillNameEdit.text())
        return spillname
        

    def retardationDueOilType(self):
        oilType = self.OilTypeSpinBox.value()
        if oilType == 1.0:
            self.retardation = 5.6
        if oilType == 2.0:
            self.retardation = 4.2
        if oilType == 3.0:
            self.retardation = 2.8
        if oilType == 4.0:
            self.retardation = 1.4
        if oilType == 5.0:
            self.retardation = 0.0

        
    # Cannot be performed inside retardationDueOilType because that is done when N or S clicking, when no DLx, DLy, DLcon have been input yet.
    # needs to be called after data has been uploaded and processed.
    def x0y0DueSinkingRetardation(self):
        B = [max(self.DLcon[vld]) for vld in xrange(len(self.st))]
        C = [self.DLcon[vld].index(max(self.DLcon[vld])) for vld in xrange(len(self.st))]
        x0new = [self.DLx[vld][C[vld]] for vld in xrange(len(self.st))]
        y0new = [self.DLy[vld][C[vld]] for vld in xrange(len(self.st))]
        distX = [x0new[vld] - self.x0 for vld in xrange(len(self.st))]
        distY = [y0new[vld] - self.y0 for vld in xrange(len(self.st))]
        oilType = self.OilTypeSpinBox.value()
        if oilType == 1.0:
            sunkx0 = [(self.x0 + (7.0*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (7.0*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)
        if oilType == 2.0:
            sunkx0 = [(self.x0 + (5.6*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (5.6*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)
        if oilType == 3.0:
            sunkx0 = [(self.x0 + (4.2*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (4.2*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)
        if oilType == 4.0:
            sunkx0 = [(self.x0 + (2.8*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (2.8*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)
        if oilType == 5.0:
            sunkx0 = [(self.x0 + (1.4*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (1.4*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)
        if oilType == 6.0:
            sunkx0 = [(self.x0 + (0.0*(distX[vld]/8.0))) for vld in xrange(len(self.st))]
            sunky0 = [(self.y0 + (0.0*(distY[vld]/8.0))) for vld in xrange(len(self.st))]
            self.sunkx0 = sum(sunkx0)/sum(B)
            self.sunky0 = sum(sunky0)/sum(B)


            
#__________________________________________________________________________________________________________________________
    # Methods for the Canvas and displays:


    def zoomInArea(self):
        self.canvas.setMapTool(self.toolZoomIn)


    def zoomIn(self):
        self.canvas.setMapTool(self.toolZoomIn)


    def zoomOut(self):
        self.canvas.setMapTool(self.toolZoomOut)


    def pan(self):
        self.canvas.setMapTool(self.toolPan)


    def captureCoords(self):
        self.canvas.setMapTool(self.toolCaptureCoordinates)
        

    def addRasterImage(self):
        """add a (user-selected) raster image"""

        s = QSettings()
        oldValidation = s.value( "/Projections/defaultBehaviour" )
        s.setValue( "/Projections/defaultBehaviour", "useGlobal" )

        fileName = 'Desktop'


        dir = os.path.dirname(fileName) if fileName is not None else "."
        formats = ["*.%s" % "PNG"]  
        
        file = unicode(QFileDialog.getOpenFileName(self, "SOSim - Choose Raster Image to Import", dir, "PNG (%s)" % " ".join(formats)))
        if file:
            layer = QgsRasterLayer(file, fileName)

            if not layer.isValid():
                return

            # add layer to the registry

            layer.setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )

            s.setValue( "Projections/defaultBehaviour", oldValidation )
            QgsMapLayerRegistry.instance().addMapLayer(layer)

            # # set extent to the extent of layer
            self.canvas.setExtent(layer.extent())
            
            self.layers.insert(0, QgsMapCanvasLayer(layer))
            
            self.canvas.setLayerSet(self.layers)
            self.MyWorldLayer(ext = False)
    
    def addRasterImage1(self):
        x = self.lon0    
        y = self.lat0
        self.layers = []

        fileName = 'Desktop' 

        dir = os.path.dirname(fileName) if fileName is not None else "."
        formats = ["*.%s" % "PNG"]  
        
        file = myfile_list[k_zt]  #raster image to load onto vector image


        if file:
            layer = QgsRasterLayer(file, fileName)


            if not layer.isValid():
                return

            ext = layer.extent()
            xmin = ext.xMinimum()
            ymin = ext.yMinimum()
            xmax = ext.xMaximum()
            ymax = ext.yMaximum()


            # add layer to the registry
            QgsMapLayerRegistry.instance().addMapLayer(layer)

            # # set extent to the extent of layer
            self.canvas.setExtent(QgsRectangle(xmin,ymin,xmax,ymax))                       
            self.layers.insert(0, QgsMapCanvasLayer(layer))

            print "layers number:",len(self.layers)
            self.canvas.setLayerSet(self.layers)
            #self.canvas.refresh()
            self.MyWorldLayer(ext = True)


    def MyWorldLayer(self, ext=bool):
        """add the base map layer and zoom to its extent when needed"""

        s = QSettings()
        oldValidation = s.value( "/Projections/defaultBehaviour" )
        s.setValue( "/Projections/defaultBehaviour", "useGlobal" )

        file = SOSimPath + "Data/world-final.shp"
        print(file)
        fileName = "world-final"

        # create layer
        layer = QgsVectorLayer(file, fileName, "ogr")

        if not layer.isValid():
            return

        layer.setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )
        s.setValue( "Projections/defaultBehaviour", oldValidation )

        # Change the color of the layer to green:
        symbols = layer.rendererV2().symbols()
        symbol = symbols[0]
        symbol.setColor(QColor.fromRgb(113,170,0))

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer);

        # set extent to the extent of our layer
        if ext == True:
            self.canvas.setExtent(layer.extent())
        if ext == False:
            rect = QgsRectangle(float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max))
            self.canvas.setExtent(rect)

        # set the canvas layer set:
        self.layers.insert(0, QgsMapCanvasLayer(layer))
        self.canvas.setLayerSet(self.layers)
        

    def MyWorldLayerGlobalCanvas(self):
        """add the base map layer and zoom to its extent"""

        s = QSettings()
        oldValidation = s.value( "/Projections/defaultBehaviour" )
        s.setValue( "/Projections/defaultBehaviour", "useGlobal" ) 

        file = SOSimPath + "Data/world-final.shp"        
        fileName = "world-final"

        # create layer
        layer = QgsVectorLayer(file, fileName, "ogr")

        if not layer.isValid():
            return

        layer.setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )
        s.setValue( "Projections/defaultBehaviour", oldValidation )

        # Change the color of the layer to green:
        symbols = layer.rendererV2().symbols()
        symbol = symbols[0]
        symbol.setColor(QColor.fromRgb(113,170,0))

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer)

        # set extent to all extension:
        self.globalcanvas.setExtent(layer.extent())

        # set the global canvas layer set:
        gl = QgsMapCanvasLayer(layer)
        globallayers = [gl]
        self.globalcanvas.setLayerSet(globallayers)


    def MyGraticuleLayer(self):
        """add the graticule layer to the main canvas"""

        s = QSettings()
        oldValidation = s.value( "/Projections/defaultBehaviour" )
        s.setValue( "/Projections/defaultBehaviour", "useGlobal" )

        file = SOSimPath + "/Data/WorldGraticule.shp"
        fileName = "WorldGraticule"

        # create layer
        layer = QgsVectorLayer(file, fileName, "ogr")

        if not layer.isValid():
            return

        layer.setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )
        s.setValue( "Projections/defaultBehaviour", oldValidation )

        # Change the color of the layer to green:
        symbols = layer.rendererV2().symbols()
        symbol = symbols[0]
        symbol.setColor(QColor.fromRgb(250,250,250))

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer);

        # set extent to the extent of our layer
        rect = QgsRectangle(float(self.x_min), float(self.y_min), float(self.x_max), float(self.y_max))
        self.canvas.setExtent(rect)

        # set the canvas layer set:
        self.layers.insert(0, QgsMapCanvasLayer(layer))
        self.canvas.setLayerSet(self.layers)


    def MyRasterLoad(self, file, fileName):
        """Load any raster layer specifying its file string"""

        s = QSettings()
        oldValidation = s.value( "/Projections/defaultBehaviour" )
        s.setValue( "/Projections/defaultBehaviour", "useGlobal" )
        layer = QgsRasterLayer(file, fileName)

        if not layer.isValid():
            return


        layer.setCrs( QgsCoordinateReferenceSystem(4326, QgsCoordinateReferenceSystem.EpsgCrsId) )
        s.setValue( "Projections/defaultBehaviour", oldValidation )

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer)

        # set extent to the extent of layer
        self.canvas.setExtent(layer.extent())

        self.layers.insert(0, QgsMapCanvasLayer(layer))
        self.canvas.setLayerSet(self.layers)

    def MyLegendLoad(self, file, fileName):
        """Load the corresponding legend of any raster layer specifying its file string"""
        # create layer
        layer = QgsRasterLayer(file, fileName)

        if not layer.isValid():
            return

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer)

        # set extent to the extent of our layer
        self.legendcanvas.setExtent(layer.extent())
        self.legendcanvas.refresh()

        # set the legend canvas layer set
        cl = QgsMapCanvasLayer(layer)
        layers = [cl]
        self.legendcanvas.setLayerSet(layers)
        
    def MyLegendLoad2(self, file, fileName):
        """Load the corresponding legend of any raster layer specifying its file string"""
        # create layer
        layer = QgsRasterLayer(file, fileName)

        if not layer.isValid():
            return

        # add layer to the registry
        QgsMapLayerRegistry.instance().addMapLayer(layer)

        # set extent to the extent of our layer
        self.outlegendcanvas.setExtent(layer.extent())
        self.outlegendcanvas.refresh()

        # set the legend canvas layer set
        cl = QgsMapCanvasLayer(layer)
        layers = [cl]
        self.outlegendcanvas.setLayerSet(layers)
         
    #_________________________________________________________________________________________________________________________


    @pyqtSignature("")
    def on_SpillE_clicked(self):
        class NoLongitudeError(Exception): pass
        class LongitudeUnitsError(Exception): pass
        lon0 = unicode(self.LongSpillEdit.text())
        print "Lon0 IS: %s" % lon0
        try:
            if len(lon0) == 0:
                raise NoLongitudeError, ("The longitude edit line may not be empty")
            if "." not in lon0 or "'" in lon0:
                raise LongitudeUnitsError, ("The longitude must be written in decimal units")
        except NoLongitudeError, e:
            QMessageBox.warning(self, "SOSim - No Longitude in edit line", unicode(e))
            self.EWSpillBox.reset()
            self.LongSpillEdit.setFocus()
            return
        except LongitudeUnitsError,e:
            QMessageBox.warning(self, "SOSim - Longitude units error", unicode(e))
            self.EWSpillBox.reset()
            self.LongSpillEdit.selectAll()
            self.LongSpillEdit.setFocus()
            return

        print "__________________SpillE_________________________"
        self.lon0 = lon0

        self.showSpillSiteCanvas(4)

        
    @pyqtSignature("")
    def on_SpillW_clicked(self):
        class NoLongitudeError(Exception): pass
        class LongitudeUnitsError(Exception): pass
        lon0 = unicode(self.LongSpillEdit.text())
        print "Lon0 is: %s" % lon0
        try:
            if len(lon0) == 0:
                raise NoLongitudeError, ("The longitude edit line may not be empty")
            if "." not in lon0 or "'" in lon0:
                raise LongitudeUnitsError, ("The longitude must be written in decimal units")
        except NoLongitudeError, e:
            QMessageBox.warning(self, "SOSim - No Longitude in edit line", unicode(e))
            self.EWSpillBox.reset()
            self.LongSpillEdit.setFocus()
            return
        except LongitudeUnitsError,e:
            QMessageBox.warning(self, "SOSim - Longitude units error", unicode(e))
            self.EWSpillBox.reset()
            self.LongSpillEdit.selectAll()
            self.LongSpillEdit.setFocus()
            return
        lon0 = float(lon0)
        self.lon0 = -lon0

        self.showSpillSiteCanvas(4)   

        print "__________________SpillW_________________________"
        
        
    @pyqtSignature("")
    def on_SpillN_clicked(self):
        class NoLatitudeError(Exception): pass
        class LatitudeUnitsError(Exception): pass
        lat0 = unicode(self.LatSpillEdit.text())

        print "Lat0 is: %s" % lat0        
        try:
            if len(lat0) == 0:
                raise NoLatitudeError, ("The latitude edit line may not be empty")
            if "." not in lat0 or "'" in lat0:
                raise LatitudeUnitsError, ("The latitude must be written in decimal units")
        except NoLatitudeError, e:
            QMessageBox.warning(self, "SOSim - No Latitude in edit line", unicode(e))
            self.NSSpillgroupBox.reset()
            self.LatSpillEdit.setFocus()
            return
        except LatitudeUnitsError,e:
            QMessageBox.warning(self, "SOSim - Latitude units error", unicode(e))
            self.NSSpillgroupBox.reset()
            self.LatSpillEdit.selectAll()
            self.LatSpillEdit.setFocus()
            return           
        print "__________________SpillN_________________________"        
        self.lat0 = float(lat0)
        
        self.showSpillSiteCanvas(4)
        self.drawSpill0()
        self.TitleEdit.setText("MODEL SET UP FOR SPILL %s" % self.spillName().upper())
            
    @pyqtSignature("")
    def on_SpillS_clicked(self):
        class NoLatitudeError(Exception): pass
        class LatitudeUnitsError(Exception): pass
        lat0 = unicode(self.LatSpillEdit.text())
        try:
            if len(lat0) == 0:
                raise NoLatitudeError, ("The latitude edit line may not be empty")
            if "." not in lat0 or "'" in lat0:
                raise LatitudeUnitsError, ("The latitude must be written in decimal units")
        except NoLatitudeError, e:
            QMessageBox.warning(self, "SOSim - No Latitude in edit line", unicode(e))
            self.NSSpillgroupBox.reset()
            self.LatSpillEdit.setFocus()
            self.SpillS.reset()
            return
        except LatitudeUnitsError,e:
            QMessageBox.warning(self, "SOSim - Latitude units error", unicode(e))
            self.NSSpillgroupBox.reset()
            self.LatSpillEdit.selectAll()
            self.LatSpillEdit.setFocus()
            return           
        print "__________________SpillS_________________________"        
        lat0 = float(lat0)
        self.lat0 = -lat0
        self.showSpillSiteCanvas(4) 
        self.drawSpill0()

        self.TitleEdit.setText("Model Set Up for Spill %s" % self.spillName())
        


    def drawSpill0(self):
        if self.submerged.isChecked():
            x = self.lon0    
            y = self.lat0
            x=float(x)  
            y=float(y)  

            if len( self.markers ) > 0:
                self.canvas.scene().removeItem( self.markers[ 0 ] )
                del self.markers[ 0 ]
            self.marker = QgsVertexMarker( self.canvas )
            self.marker.setCenter( QgsPoint(x,y))
            self.marker.setIconType( 2 )
            self.marker.setIconSize( 12 )
            self.marker.setColor( QColor( 255,0,48,255 ) )
            self.marker.setPenWidth ( 2 )
            self.markers.append( self.marker )
            # For the global, small canvas:
            if len(self.markersglobal) > 0:
                self.globalcanvas.scene().removeItem( self.markersglobal[0] )
                del self.markersglobal[0]
            self.markerglobal = QgsVertexMarker( self.globalcanvas )
            self.markerglobal.setCenter(QgsPoint(x, y))
            self.markerglobal.setIconType(2) 
            self.markerglobal.setIconSize(6)
            self.markerglobal.setColor(QColor(255,0,48,255 ))
            self.markerglobal.setPenWidth(1.2)
            self.markersglobal.append(self.markerglobal)
        if self.sunken.isChecked():
            x = self.lon0    
            y = self.lat0
            x=float(x) 
            y=float(y) 

            if len( self.markers ) > 0:
                self.canvas.scene().removeItem( self.markers[ 0 ] )
                del self.markers[ 0 ]
            self.marker = QgsVertexMarker( self.canvas )
            self.marker.setCenter( QgsPoint(x,y))
            self.marker.setIconType( 2 )
            self.marker.setIconSize( 12 )
            self.marker.setColor( QColor( 255,0,48,255 ) )
            self.marker.setPenWidth ( 2 )
            self.markers.append( self.marker )
            # For the global, small canvas:
            if len(self.markersglobal) > 0:
                self.globalcanvas.scene().removeItem( self.markersglobal[0] )
                del self.markersglobal[0]
            self.markerglobal = QgsVertexMarker( self.globalcanvas )
            self.markerglobal.setCenter(QgsPoint(x, y))
            self.markerglobal.setIconType(2) 
            self.markerglobal.setIconSize(6)
            self.markerglobal.setColor(QColor(255,0,48,255 ))
            self.markerglobal.setPenWidth(1.2)
            self.markersglobal.append(self.markerglobal)

    def showSpillSiteCanvas(self, scale):
        x = self.lon0
        y = self.lat0
        rect = QgsRectangle(float(x)-scale,float(y)-scale,float(x)+scale,float(y)+scale) 
        mc = self.canvas
        mc.setExtent(rect)
        mc.refresh()
        mglobal = self.globalcanvas
        mglobal.setExtent(rect)
        mglobal.refresh()


    def updateCoordsDisplay(self, point, button):
        mypoint = QString(str(point.x())+ "," + str(point.y()))
        self.lineEdit.setText(point)


    def set_x0(self):
        self.x0 = (LatLongConverter(self.lon0, self.lat0))[0]
        print "x0: %s" % self.x0
        return self.x0


    def set_y0(self):
        self.y0 = (LatLongConverter(self.lon0, self.lat0))[1]
        print "y0: %s" % self.y0
        return self.y0


    def set_zone0(self):
        self.zone0 = (LatLongConverter(self.lon0, self.lat0))[2]
        return self.zone0

      
#__________________________________________________________________________________________________________________________
    # Methods for the Spill Start Time Layout:

    @pyqtSignature("QTime")
    def on_SpillStartHourEdit_timeChanged(self):
        SpillStartHour = self.SpillStartHourEdit.time()
        print SpillStartHour
        self.SpillEndHourEdit.setTime(SpillStartHour)
        self.SampHourEdit.setTime(SpillStartHour)
        self.CalcHourEdit.setTime(SpillStartHour)
        return SpillStartHour   

    @pyqtSignature("QDate")
    def on_SpillStartDateEdit_dateChanged(self):
        spilldate = self.SpillStartDateEdit.date()

        self.SampDateEdit.setDate(spilldate)
        self.CalcDateEdit.setDate(spilldate)
        self.SpillEndDateEdit.setDate(spilldate)

        
        return spilldate


    def SetTimeZero(self):
#        HourZero = self.on_SpillStartHourEdit_timeChanged()##        DateZero = self.on_SpillStartDateEdit_dateChanged()
        HourZero=self.SpillStartHourEdit.time()
        hours = HourZero.hour()
        minutes = HourZero.minute()
        HourZero1 = hours + (minutes/60.0)

        DateZero=self.SpillStartDateEdit.date()
        months = int(DateZero.month())
        days = int(DateZero.day())
        years = int(DateZero.year())
        DateZero1 = [int(years), int(months), int(days)] 

        DateZero1.append(HourZero1)
        TimeZero = DateZero1
        
        return TimeZero
        #__________________________________________________________________________________________________________________________

    # Methods for the Spill End Time Layout:
    @pyqtSignature("QTime")
    def on_SpillEndHourEdit_timeChanged(self):
        SpillEndHour = self.SpillEndHourEdit.time()
        hours = SpillEndHour.hour()
        minutes = SpillEndHour.minute()
        HourZero = hours + (minutes/60.0)
        print "HourEnd", HourZero
        return HourZero
        
    @pyqtSignature("QDate")
    def on_SpillEndDateEdit_dateChanged(self):
        spilldate = self.SpillEndDateEdit.date()
        months = int(spilldate.month())
        days = int(spilldate.day())
        years = int(spilldate.year())
        DateZero = [int(years), int(months), int(days)]
        print "Spill end:", DateZero
        return DateZero
  
    def SetTimeEnd(self):
        HourZero = self.on_SpillEndHourEdit_timeChanged()
        DateZero = self.on_SpillEndDateEdit_dateChanged()
        DateZero.append(HourZero)
        TimeZero = DateZero
        print "TIME End IS %s" %TimeZero
        return TimeZero
       

#__________________________________________________________________________________________________________________________
    # Methods for the Sampling Procedure Layout:


    # FOR USER TO UPLOAD DATA:
    @pyqtSignature("")
    def on_UploadCampaignButton_clicked(self):
        if not (self.SpillE.isChecked() or self.SpillW.isChecked()) or not(self.SpillN.isChecked() or self.SpillS.isChecked()):
            QMessageBox.warning(self, "SOSim - Undefined longitude or latitude", "Please define the direction of the spill coordinates before uploading a sampling campaign")
            self.LatSpillEdit.selectAll()
        else:
            class NoSuchFileError(Exception): pass
        
            # for control of number of points per campaign and number and time of campaigns:
            index = self.CampaignNumberSpinBox.value() - 1
            self.allCampaignIndices.append(index)
            print "Campaign indexes' on file are: %s" % self.allCampaignIndices

            QMessageBox.information(self, "SOSim - About Campaign Data File",
                                    "You should upload the corresponding file in correct format. (See User Manual for format. Please press the ok button)")
        
            try:
                DataList = []
                filename = 'Campaigns'
                dir = os.path.dirname(filename) \
                      if filename is not None else "."
                print dir
                formats = ["*.%s" % "*"]  
                fname = unicode(QFileDialog.getOpenFileName(self,
                                "SOSim - Choose Campaign Data File", dir,
                                "All files (%s)" % " ".join(formats)))

                print fname
                self.ourinformation['CampaignButton'].append(fname)
                self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))


                print self.ourinformation['CampaignButton']
                print "______________________________"

                tmp=0
                if fname:
                    tmp=tmp+1
                    #print tmp
                #print fname 
                
                if fname:
                    fieldsep = "\t"
                    F = open(fname, "r")
                    Lines = F.readlines()
                for i in Lines:
                    
                    L1 = string.strip(i)
                    #print(L1)
                    Record = L1.split(',')
                    #print(Record)
                    DataList.append(Record)
                F.close()
                #print(DataList)
                DataList.pop(0)
                #print(DataList)     
                DL = transpose(array(DataList))
                #print(DL)
                LonWGS = map(float,DL[0])
                print "LonWGS: %s" % LonWGS
                LatWGS = map(float,DL[1])
                print "LatWGS: %s" % LatWGS
                DLcon = map(float,DL[2]) 
                
                LatLongUTM = []
                i = 0
                while i < len(LonWGS):
                    LatLongUTM.append(LatLongConverter(LonWGS[i], LatWGS[i]))
                    i += 1
                
                conValues = []
                for s in DLcon:
                    if s == 0.0:
                        conValue = 0.1/100.0
                    else:
                        conValue = s
                    conValues.append(conValue)

                self.DLcon[index] = map(float, conValues)
                LatLongUTM = transpose(numpy.array(LatLongUTM))
                self.DLx[index] = map(float,LatLongUTM[0])
                self.DLy[index] = map(float,LatLongUTM[1])
                self.DLzone[index] = map(float,LatLongUTM[2])

                # for setting up the final data lists with all the campaigns, units in km and zone issue addressed:
                a = index + 1
            
                more = QMessageBox.question(self, "SOSim - About Sampling Campaigns Data Files",
                                     "Data for Campaign number %i has been uploaded and processed.\
                                     Do you want to upload more sampling campaigns now?" % a,
                                     QMessageBox.Yes|QMessageBox.No)
                if more == QMessageBox.Yes:
                    self.addToMultipleSamplingTimes(index)
                    self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
                    
                if more == QMessageBox.No:
                	self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
                	self.UploadCampaignButton.setEnabled(False)
                	self.addToMultipleSamplingTimes(index)
	                i=0
	                while i < 10:
	                    for k in self.DLcon:
	                        if k == [0.0]:
	                            self.DLcon.remove(k)
	                    for k in self.DLx:
	                        if k == [0.0]:
	                            self.DLx.remove(k)
	                    for k in self.DLy:
	                        if k == [0.0]:
	                            self.DLy.remove(k)
	                    for k in self.DLzone:
	                        if k == [0.0]:
	                            self.DLzone.remove(k)
	                    i += 1
	                campsize = []
	                i=0
	                while i < len(self.DLcon):
	                    campsize.append(len(self.DLcon[i]))
	                    i += 1
                    
                
	                DLxJoin = []
	                DLyJoin = []
	                DLzoneJoin = []
	                k=0
	                while k < len(campsize):
	                    DLxJoin = DLxJoin + self.DLx[k]
	                    DLyJoin = DLyJoin + self.DLy[k]
	                    DLzoneJoin = DLzoneJoin + self.DLzone[k]
	                    k +=1
	                
	                # Adjusting points out of the working (the zone with most sampling points) UTM zone:
	                zonearray = numpy.array(DLzoneJoin)
	                average = mean(zonearray)
	                
	                if average is not int:
	                    good_zone = round(average)
	                    reference_point_index = DLzoneJoin.index(good_zone)
	                    reference_point_coords = UTMConverter((DLxJoin[reference_point_index])*1000.0, (DLyJoin[reference_point_index])*1000.0, good_zone)
	                    for bad_zone in DLzoneJoin:
	                        if bad_zone != good_zone:
	                            bad_point_index = DLzoneJoin.index(bad_zone)
	                            bad_point_longitude = (UTMConverter((DLxJoin[bad_point_index])*1000.0, (DLyJoin[bad_point_index])*1000, bad_zone))[0] # Latitude does not change
	                            auxiliar_point_longitude = reference_point_coords[0] + (reference_point_coords[0] - bad_point_longitude)
	                            auxiliar_point_latitude = reference_point_coords[1]
	                            auxiliar_point_easting = (LatLongConverter(auxiliar_point_longitude, auxiliar_point_latitude))[0]
	                            
	                            fake_bad_point_easting = DLxJoin[reference_point_index] + (DLxJoin[reference_point_index] - auxiliar_point_easting)
	                            DLxJoin[bad_point_index] = fake_bad_point_easting
	                            DLzoneJoin[bad_point_index] = good_zone
	                        
	                # Convert back to list of lists per campaign:
	                m = 0
	                while m < len(campsize):
	                    self.DLx[m] = DLxJoin[int(sum(campsize[0:m])):int(campsize[m] + sum(campsize[0:m]))]
	                    self.DLzone[m] = DLzoneJoin[int(sum(campsize[0:m])):int(campsize[m] + sum(campsize[0:m]))]
	                    m += 1

                self.RunButton.setEnabled(True)
                self.CalibrateButton.setEnabled(True)
                self.RecalcButton.setEnabled(True)
            except NoSuchFileError, e:
	            QMessageBox.warning(self, "No such file was found in disk, please try again", unicode(e))
	            
	            self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
	            return
    
  
    @pyqtSignature("")        
    def on_RemoveCampaignButton_clicked(self):
        index = self.CampaignNumberSpinBox.value() - 1
        #class NoDataToDeleteError(Exception): pass
        a = index + 1
        more = QMessageBox.question(self, "SOSim - About Sampling Campaigns Data Files",
                                 "Are you sure you want to delete data uploaded for sampling campaign number %i?" % a,
                                 QMessageBox.Yes|QMessageBox.No)
        if self.DLcon[index] != 0.0:
            more
            if more == QMessageBox.Yes:

                self.ourinformation['CampaignButton'].pop()
                self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
                #print self.ourinformation['CampaignButton']
                #print len(self.ourinformation['CampaignButton'])
                print "__________________________________________on_RemoveCampaignButton_clicked____________________________________________________________________"

                self.UploadCampaignButton.setEnabled(True)
                self.deleteFromMultipleSamplingTimes(index)
                self.DLx.pop(index)
                self.DLx.insert(index,[0.0])
                self.DLy.pop(index)
                self.DLy.insert(index,[0.0])
                self.DLcon.pop(index)
                self.DLcon.insert(index,[0.0])
                
            if more == QMessageBox.No:
                QMessageBox.information(self, "SOSim - About Sampling Campaigns Data Files", "No sampling data has been deleted.")
        else:
            QMessageBox.warning(self, "SOSim - Deletion error", "There is no such campaign number %i in file. Cannot delete or deleted already." % a)
            
            self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))


    @pyqtSignature("")
    def on_OpenCampaignButton_clicked(self):
        class NoSuchFileError(Exception): pass
        try:
            filename = 'Desktop'
            dir = os.path.dirname(filename) \
                  if filename is not None else "."
            formats = ["*.%s" % "txt"]  
            fname = unicode(QFileDialog.getOpenFileName(self,
                            "SOSim - Choose Existing Calibration File", dir,
                            "Campaign files (%s)" % " ".join(formats)))
            if fname:
                shutil.copy(fname, PlayPath + "/my_LF.txt")
                shutil.copy(fname.replace(".txt", "") + "GammaCombIndex.txt", PlayPath + "/my_GammaCombIndex.txt")
                shutil.copy(fname.replace(".txt", "") + "MaxST.txt", PlayPath + "/my_MaxST.txt")
                shutil.copy(fname.replace(".txt", "") + "DLx.txt", PlayPath + "/my_DLx.txt")
                shutil.copy(fname.replace(".txt", "") + "DLy.txt", PlayPath + "/my_DLy.txt")
                shutil.copy(fname.replace(".txt", "") + "DLcon.txt", PlayPath + "/my_DLcon.txt")
                QMessageBox.information(self, "SOSim - Use Existing Campaign", "Calibration '%s' is now in use." % fname)
                self.importedCalibration = True
                self.RecalcButton.setEnabled(True)
                self.RunButton.setEnabled(False)
                self.CalibrateButton.setEnabled(False)
        except NoSuchFileError, e:
            QMessageBox.warning(self, "No such file was found in disk, please try again", unicode(e))
            
            self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
            return




#__________________________________________________________________________________________________________________________
    # Methods for Submerged Hydrodynamic campaign(s)


    # FOR USER TO UPLOAD Submerged Hydrodynamic:
    @pyqtSignature("")
    def on_UploadSubmergedHydroButton_clicked(self):
        if not (self.SpillE.isChecked() or self.SpillW.isChecked()) or not(self.SpillN.isChecked() or self.SpillS.isChecked()):
        	if self.submerged.isChecked():
        		QMessageBox.warning(self, "SOSim - Undefined longitude or latitude", "Please define the direction of the spill coordinates before uploading a hydrodynamic model outputs")
        		self.LatSpillEdit.selectAll()
        	if self.sunken.isChecked():
        		QMessageBox.warning(self, "SOSim - Undefined longitude or latitude", "Please define the direction of the spill coordinates before uploading a bathymetric data file")
        		self.LatSpillEdit.selectAll()
        else:
            class NoSuchFileError(Exception): pass
            # for control of number of points per campaign and number and time of campaigns:
            index = self.CampaignNumberSpinBox.value() - 1
            self.allCampaignIndices.append(index)
            print "Campaign indexes' on file are: %s" % self.allCampaignIndices

            if self.submerged.isChecked():
            	QMessageBox.information(self, "SOSim - About Hydrodynamic Model File",
            		"You should upload the corresponding file in correct format. (Please press the OK button).")
            if self.sunken.isChecked():
            	QMessageBox.information(self, "SOSim - About Bathymetry File",
            		"You should upload the corresponding file in correct format. (See User Manual for format. Please press the OK button).")
        
            try:
                DataList = []
                filename = 'Campaigns'
                dir = os.path.dirname(filename) \
                      if filename is not None else "."
                formats = ["*.%s" % "*"]  
                fname = unicode(QFileDialog.getOpenFileName(self,
                                "SOSim - Choose Campaign Data File", dir,
                                "All files (%s)" % " ".join(formats)))

                self.ourinformation['HydroButton'] = fname
                self.UploadSubmergedHydroButton.setEnabled(False)

                
                self.RunButton.setEnabled(True)
                self.CalibrateButton.setEnabled(True)
                self.RecalcButton.setEnabled(True)
 
            except NoSuchFileError, e:
                QMessageBox.warning(self, "No such file was found in disk, please try again", unicode(e))
                self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
                return
    
  
    @pyqtSignature("")        
    def on_RemoveSubmergedButton_clicked(self):
        index = self.CampaignNumberSpinBox.value() - 1
        
        a = index + 1
        if self.submerged.isChecked():
        	more = QMessageBox.question(self, "SOSim - About Hydrodynamic Data Files",
        		"Are you sure you want to delete data uploaded for Hydrodynamic Data number %i?" % a,
        		QMessageBox.Yes|QMessageBox.No)
        if self.sunken.isChecked():
        	more = QMessageBox.question(self, "SOSim - About Bathymetry Data Files",
        		"Are you sure you want to delete data uploaded for Bathymetry Data number %i?" % a,
        		QMessageBox.Yes|QMessageBox.No)
        if self.DLcon[index] != 0.0:
            more
            if more == QMessageBox.Yes:
            	self.UploadSubmergedHydroButton.setEnabled(True)

                self.ourinformation['HydroButton'] = []

                self.deleteFromMultipleSamplingTimes(index)
                self.DLx.pop(index)
                self.DLx.insert(index,[0.0])
                self.DLy.pop(index)
                self.DLy.insert(index,[0.0])
                self.DLcon.pop(index)
                self.DLcon.insert(index,[0.0])
                
            if more == QMessageBox.No:
                QMessageBox.information(self, "SOSim - About Hydrodynamic Data Files", "No sampling data has been deleted.")
        else:
            QMessageBox.warning(self, "SOSim - Deletion error", "There is no such campaign number %i in file. Cannot delete or deleted already." % a)
            # self.CampaignNumberSpinBox.setFocus()
            self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))


#__________________________________________________________________________________________________________________________
    # Methods for the Sampling Time Layout:    

    @pyqtSignature("QTime")
    def on_SampHourEdit_timeChanged(self):
        samphour = self.SampHourEdit.time()
        hours = samphour.hour()
        minutes = samphour.minute()
        HourSamp = hours + (minutes/60.0)
        print "HourSamp:", HourSamp
        return HourSamp
        

    @pyqtSignature("QDate")
    def on_SampDateEdit_dateChanged(self):
        samplingdate = self.SampDateEdit.date()
        months = samplingdate.month()      
        days = samplingdate.day()
        years = samplingdate.year()
        DateSamp = [years, months, days]
        print "DateSamp:", DateSamp
        return DateSamp
  

    def SetSampTime(self):  
        HourSamp = self.on_SampHourEdit_timeChanged()
        DateSamp = self.on_SampDateEdit_dateChanged()
        DateSamp.append(HourSamp)
        DateSamp = numpy.array(DateSamp)
        #print "DateSampWithHour:", DateSamp
        TimeZero = numpy.array(self.SetTimeZero())
        print "Verify TimeZero passed ok:", TimeZero
        stlist = DateSamp - TimeZero
        #print "Verify sampletime ok to check st final:", stlist
        
        daysinmonth = (calendar.monthrange(int(TimeZero[0]),int(TimeZero[1])))[1]
        a=1
        daysinyear = 0
        while a <= 12:
            daysinyear = daysinyear + (calendar.monthrange(int(TimeZero[0]),int(a)))[1]
            a += 1
        if stlist[3] < 0.0:
            stlist[2] = stlist[2]-1.0
            stlist[3] = 24.0 + stlist[3]
        if stlist[2] < 0.0:
            stlist[1] = stlist[1]-1.0
            stlist[2] = daysinmonth + stlist[2]
        if stlist[1] < 0.0:
            stlist[0] = stlist[0]-1.0
            stlist[1] = 12.0 + stlist[1]
        
        if stlist[0] < 0.0:
            QMessageBox.critical(self, "SOSim - Dates error", "Sampling campaign dates must follow a spill event.")
            self.SampDateEdit.selectAll()
            self.SampDateEdit.setFocus()
        else:
            st = (daysinyear * stlist[0]) + (daysinmonth * stlist[1]) + stlist[2] + (stlist[3]/24.0)
            if st > daysinyear:
                QMessageBox.critical(self, "SOSim - Dates error", "Revise spill and sampling years, delete sampling campaign and correct sampling date before uploading a new one.")
            
        return st


    def SetDurationTime(self):  
        TimeZero = numpy.array(self.SetTimeZero())
        
        if self.continuous.isChecked():
            TimeEnd = numpy.array(self.SetTimeEnd())
        else:
            TimeEnd = TimeZero
        print "Verify TimeZero passed ok:", TimeZero
        print "Verify TimeEnd passed ok:", TimeEnd
        drlist = TimeEnd - TimeZero
        daysinmonth = (calendar.monthrange(int(TimeZero[0]),int(TimeZero[1])))[1]
        a=1
        daysinyear = 0
        while a <= 12:
            daysinyear = daysinyear + (calendar.monthrange(int(TimeZero[0]),int(a)))[1]
            a += 1
        if drlist[3] < 0.0:
            drlist[2] = drlist[2]-1.0
            drlist[3] = 24.0 + drlist[3]
        if drlist[2] < 0.0:
            drlist[1] = drlist[1]-1.0
            drlist[2] = daysinmonth + drlist[2]
        if drlist[1] < 0.0:
            drlist[0] = drlist[0]-1.0
            drlist[1] = 12.0 + drlist[1]
        if drlist[0] < 0.0:
            QMessageBox.critical(self, "SOSim - Dates error", "Sampling campaign dates must follow a spill event.")
            self.SpillEndDateEdit.selectAll()
            self.SpillEndDateEdit.setFocus()
        else:
            dr = (daysinyear * drlist[0]) + (daysinmonth * drlist[1]) + drlist[2] + (drlist[3]/24.0)
            if dr > daysinyear:
                QMessageBox.critical(self, "SOSim - Dates error", "Revise spill and sampling years, delete sampling campaign and correct sampling date before uploading a new one.") 

        return dr

    def addToMultipleSamplingTimes(self, index):
        SampTime = self.SetSampTime()
        DurationTime = self.SetDurationTime()
        self.retardationDueOilType()
        if SampTime < self.retardation:
            QMessageBox.warning(self, "SOSim - Inappropriate Sampling Time", "The sampling campaign is inappropriate for the oil type and needs to be changed. It is calculated that the oil has not completed the sinking process. The campaign will be deleted.")
            index = self.CampaignNumberSpinBox.value() - 1
            self.DLx.pop(index)
            self.DLx.insert(index,[0.0])
            self.DLy.pop(index)
            self.DLy.insert(index,[0.0])
            self.DLcon.pop(index)
            self.DLcon.insert(index,[0.0])
            self.CampaignNumberSpinBox.setValue(len(self.ourinformation['CampaignButton']))
            
            self.dr.insert(index, DurationTime)
        if SampTime >= self.retardation:
            self.st.insert(index, SampTime - self.retardation) # when with retardation, included in the LF.
        print "SampTime to add:", SampTime
        print "DurationTime to add:", DurationTime
            
    def deleteFromMultipleSamplingTimes(self, index):
        self.st.pop(index)

#__________________________________________________________________________________________________________________________
    # Methods for the Area and Grid Layout:    

    @pyqtSignature("")
    def on_DefaultScaleRadioButton_clicked(self):
    	self.ScaleFrame.hide()
    	self.SureButton.hide()
        if self.submerged.isChecked():
            self.showSpillSiteCanvas(0.1)
            x = self.lon0    
            y = self.lat0
            #print "lon lat",x,y
            scale = 0.20       # degrees units
            self.x_min = float(x) - scale
            self.y_min = float(y) - scale
            self.x_max = float(x) + scale
            self.y_max = float(y) + scale
            #print self.x_min, self.x_max, self.y_min, self.y_max
            print "on_AutoSelectAreaButton_clicked"
        if self.sunken.isChecked():
            self.showSpillSiteCanvas(0.2)
            x = self.lon0    
            y = self.lat0
            #print "lon lat",x,y
            scale = 0.20       # degrees units
            self.x_min = float(x) - scale
            self.y_min = float(y) - scale
            self.x_max = float(x) + scale
            self.y_max = float(y) + scale
            #print self.x_min, self.x_max, self.y_min, self.y_max
            print "on_AutoSelectAreaButton_clicked"

        if 'SpillPlace' in self.ourinformation and self.ourinformation["SpillPlace"] == "River":
            riverselection = QMessageBox.information(self, "SOSim - Model Oil Spill Location Simulator",
                                 "Consider Defining the Scale to decrease the Modeling Area if the spill occurred in a River. The Modeling Area for a spill in a River should be smaller than the Default (<0.2) and rectangular in shape.")
            riverselection
    
    
    @pyqtSignature("")
    def on_UserDefinedScaleRadioButton_clicked(self):
        
        self.ScaleFrame.setVisible(True)
        self.SureButton.setVisible(True)
        

    @pyqtSignature("")
    def on_SureButton_clicked(self):
        print "I am sure button"
        x = self.lon0
        y = self.lat0
        lon_s = float(self.lonScaleEdit.text())
        lat_s = float(self.latScaleEdit.text())
        rect = QgsRectangle(float(x)-lon_s,float(y)-lat_s,float(x)+lon_s,float(y)+lat_s) 
        mc = self.canvas
        mc.setExtent(rect)
        mc.refresh()
        mglobal = self.globalcanvas
        mglobal.setExtent(rect)
        mglobal.refresh()
        self.x_min = float(x) - lon_s
        self.y_min = float(y) - lat_s
        self.x_max = float(x) + lon_s
        self.y_max = float(y) + lat_s

    def checkResolutionConflict(self):
                
        print "Default resolution is %s" % self.popDialog.defaultResolution

    
    def setUserSelectedModelArea(self):
        InsExtent = self.canvas.extent()
        self.x_max = InsExtent.xMaximum()
        self.x_min = InsExtent.xMinimum()
        self.y_max = InsExtent.yMaximum()
        self.y_min = InsExtent.yMinimum()
        print "X MAXIMUM OF MY CANVAS IS %s" % self.x_max
        print "X MINIMUM OF MY CANVAS IS %s" % self.x_min
        print "Y MAXIMUM OF MY CANVAS IS %s" % self.y_max
        print "Y MINIMUM OF MY CANVAS IS %s" % self.y_min


    @pyqtSignature("")
    def on_DefaultNodesRadioButton_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Model Node Setting",
                                 "Please confirm that the current grid resolution is correct, the default # of Nodes is 25",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
            self.setUserSelectedModelArea()
            self.user_nodes_x = self.popDialog.user_nodes_x #EWdefault.value()
            self.user_nodes_y = self.popDialog.user_nodes_y #NSdefault.value()
            self.checkResolutionConflict()
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - 	Model Node Setting", "Please set modeling grid resolution.")
        print self.user_nodes_x, self.user_nodes_y
        print "DefaultNodesRadioButton"

        
        self.ourinformation['xNode'] = 25
        self.ourinformation['yNode'] = 25
        

    @pyqtSignature("")
    def on_UserDefinedNodesRadioButton_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - About Model Grid resolution Setting",
                                 "Please confirm that the grid resolution is correct",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
            print "UserDefinedNodesRadioButton"
            print self.xNodesEdit.text()
            print self.yNodesEdit.text()
            self.setUserSelectedModelArea()
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - 	Model Node Setting", "Please set modeling grid resolution.")
        

    def after_xy_Defined(self):
        class NoNodesError(Exception): pass
        class TooManyNodesError(Exception): pass
        
        user_nodes_x = unicode(self.xNodesEdit.text())
        user_nodes_y = unicode(self.yNodesEdit.text())
        try:
            if len(user_nodes_x) == 0:
                raise NoNodesError, ("Please enter number of x nodes in the west-east direction")
        except NoNodesError, e:
            QMessageBox.warning(self, "No nodes number in edit line", unicode(e))
            self.xNodesEdit.setFocus()
            return
        try:
            if len(user_nodes_x) == 0:
                raise NoNodesError, ("Please enter number of y nodes in the west-east direction")
        except NoNodesError, e:
            QMessageBox.warning(self, "No nodes number in edit line", unicode(e))
            self.yNodesEdit.setFocus()
            return
        
        user_nodes_x = int(user_nodes_x)
        user_nodes_y = int(user_nodes_y)
        try:
            if user_nodes_x >= 50:
                raise TooManyNodesError, ("The number of x nodes is too large. Please keep it below 50")
        except TooManyNodesError,e:
            QMessageBox.warning(self, "Too many nodes error", unicode(e))
            self.xNodesEdit.selectAll()
            self.xNodesEdit.setFocus()
            return
        try:
            if user_nodes_y >= 50:
                raise TooManyNodesError, ("The number of y nodes is too large. Please keep it below 50")
        except TooManyNodesError,e:
            QMessageBox.warning(self, "Too many nodes error", unicode(e))
            self.yNodesEdit.selectAll()
            self.yNodesEdit.setFocus()
            return
        
        self.user_nodes_x = user_nodes_x
        self.user_nodes_y = user_nodes_y
        



    def Getinformation(self):

        # lone
        lon0 = unicode(self.LongSpillEdit.text())
        print "LON0 IS: %s" % lon0
        lon0 = float(lon0)

        if self.SpillE.isChecked():
            self.ourinformation['lon'] = lon0
        else:
            self.ourinformation['lon'] = -lon0

        # lat
        lat0 = unicode(self.LatSpillEdit.text())
        print "LAT IS: %s" % lat0
        lat0 = float(lat0)
        if self.SpillN.isChecked():
            self.ourinformation['lat'] = lat0
        else:
            self.ourinformation['lat'] = -lat0
        self.ourinformation['spillname'] = self.spillName()
        print 'spillname',self.ourinformation['spillname'] 
        # start time
        SpillStartHour = self.SpillStartHourEdit.time()
        print SpillStartHour.toString()
        hours = SpillStartHour.hour()
        minutes = SpillStartHour.minute()
        HourZero = hours + (minutes/60.0)
        
        s1 = SpillStartHour.toString(Qt.TextDate)

        spilldate = self.SpillStartDateEdit.date()
        months = int(spilldate.month())
        days = int(spilldate.day())
        years = int(spilldate.year())
        
        s2 = spilldate.toString("yyyy-MM-dd")

        self.ourinformation['starttime'] = s2 + " " + s1

        # endtime
        SpillStartHour = self.SpillEndHourEdit.time()
        hours = SpillStartHour.hour()
        minutes = SpillStartHour.minute()
        HourZero = hours + (minutes/60.0)
        
        s1 = SpillStartHour.toString(Qt.TextDate)
        
        spilldate = self.SpillEndDateEdit.date()
        months = int(spilldate.month())
        days = int(spilldate.day())
        years = int(spilldate.year())
        
        s2 = spilldate.toString("yyyy-MM-dd")

        # if 'instantaneous' in self.ourinformation and self.ourinformation['instantaneous'] == 1:
        if self.instantaneous.isChecked():
            self.ourinformation['Type'] = 'instantaneous'
            self.ourinformation['endtime'] = self.ourinformation['starttime']
        else:
            self.ourinformation['Type'] = 'continuous'
            self.ourinformation['endtime'] = s2 + " " + s1

        # oilType
        oilType = self.OilTypeSpinBox.value()
        self.ourinformation['OilType'] = oilType
        
        # Node
        if self.UserDefinedNodesRadioButton.isChecked():
            self.ourinformation['xNode'] = self.xNodesEdit.text()
            self.ourinformation['yNode'] = self.yNodesEdit.text()
        else:
            # default node is: 25
            self.ourinformation['xNode'] = 25
            self.ourinformation['yNode'] = 25

        print self.ourinformation['xNode']
        print self.ourinformation['yNode']
        # print "____________________________________"

        # VX, VY, DX, DY
        if self.UserDefineParameterButton.isChecked():
            self.ourinformation['vxmin'] = self.vxMinSpinBox.text()
            self.ourinformation['vxmax'] = self.vxMaxSpinBox.text()
            self.ourinformation['vymin'] = self.vyMinSpinBox.text()
            self.ourinformation['vymax'] = self.vyMaxSpinBox.text()
            self.ourinformation['dxmin'] = self.DxMinSpinBox.text()
            self.ourinformation['dxmax'] = self.DxMaxSpinBox.text()
            self.ourinformation['dymin'] = self.DyMinSpinBox.text()
            self.ourinformation['dymax'] = self.DyMaxSpinBox.text()
        if self.submerged.isChecked():
            self.ourinformation['vxmin'] = -3.0
            self.ourinformation['vxmax'] = 3.0
            self.ourinformation['vymin'] = -3.0
            self.ourinformation['vymax'] = 3.0
            self.ourinformation['dxmin'] = 0.01
            self.ourinformation['dxmax'] = 0.89 
            self.ourinformation['dymin'] = 0.01
            self.ourinformation['dymax'] = 0.89 
        if self.sunken.isChecked():
        	if self.RiverSpill.isChecked():
        		self.ourinformation['vxmin'] = -2.4#-2.4#-1.4#-0.5#
                self.ourinformation['vxmax'] = 2.4#2.4#1.4#0.5#2.4
                self.ourinformation['vymin'] = -67.1
                self.ourinformation['vymax'] = 67.1
                self.ourinformation['dxmin'] = 0.01#0.00001#0.01#0.00001
                self.ourinformation['dxmax'] = 0.2#0.0001#0.2#0.0001 
                self.ourinformation['dymin'] = 0.01
                self.ourinformation['dymax'] = 0.6  
        	if self.OceanSpill.isChecked():
        		self.ourinformation['vxmin'] = -3.0
        		self.ourinformation['vxmax'] = 3.0
        		self.ourinformation['vymin'] = -3.0
        		self.ourinformation['vymax'] = 3.0
        		self.ourinformation['dxmin'] = 0.01
        		self.ourinformation['dxmax'] = 0.89 
        		self.ourinformation['dymin'] = 0.01
        		self.ourinformation['dymax'] = 0.89 


        # x_min, x_max, y_min, y_max
        InsExtent = self.canvas.extent()
        self.ourinformation['x_min'] = InsExtent.xMinimum()
        self.ourinformation['x_max'] = InsExtent.xMaximum()
        self.ourinformation['y_min'] = InsExtent.yMinimum()
        self.ourinformation['y_max'] = InsExtent.yMaximum()


        # sunken button's type
        if self.sunken.isChecked():
	        if self.UTMButton.isChecked():
	        	ratio = self.textEditor1.text()
	        	self.ourinformation['SunkenUpload'] = 'UTM coord'
	        	self.ourinformation["Ratio"] = ratio
	        elif self.DecimalButton.isChecked():
	        	ratio = self.textEditor1.text()
	        	self.ourinformation['SunkenUpload'] = 'Decimal degrees'
	        	self.ourinformation["Ratio"] = ratio
	        elif self.NodataButton.isChecked():
	        	ratio = self.textEditor1.text()
	        	self.ourinformation['SunkenUpload'] = 'No Upload'
	        	self.ourinformation["Ratio"] = ratio
            
        if self.submerged.isChecked():
	        if self.OSCARButton.isChecked():
		        ratio = self.textEditor1.text()	        	
		        self.ourinformation['SubmergedType'] = 'OSCAR'
		        self.ourinformation["Ratio"] = ratio  	            
	        elif self.GNOMEButton.isChecked():
		        ratio = self.textEditor1.text()
		        self.ourinformation['SubmergedType'] = 'GNOME'
		        self.ourinformation["Ratio"] = ratio  	            
	        elif self.OtherButton.isChecked():
		        ratio = self.textEditor1.text()
		        self.ourinformation['SubmergedType'] = 'Other'
		        self.ourinformation["Ratio"] = ratio  	         
	        elif self.NodateButton.isChecked():
	            self.ourinformation['SubmergedType'] = 'Nodate'	        
        # scale
        if self.UserDefinedScaleRadioButton.isChecked():
            self.ourinformation['lonscale'] = self.lonScaleEdit.text()
            self.ourinformation['latscale'] = self.latScaleEdit.text()
        else:
            if self.sunken.isChecked():
                self.ourinformation['lonscale'] = 0.2
                self.ourinformation['latscale'] = 0.2
            if self.submerged.isChecked():
                self.ourinformation['lonscale'] = 0.1
                self.ourinformation['latscale'] = 0.1

        # spillplace
        if self.RiverSpill.isChecked():
            self.ourinformation["SpillPlace"] = "River"
    
        elif self.OceanSpill.isChecked():
            self.ourinformation["SpillPlace"] = "Ocean"
        # submerge button's type

	        #     
        if self.DefaultParameterButton_7.isChecked():
            self.ourinformation["Method"] = "Minimum"
            level = self.vyMinSpinBox_3.text()
            self.ourinformation["confidence"] = level    
        elif self.DefaultParameterButton_6.isChecked():
            self.ourinformation["Method"] = "Best"        
    

        if self.DefaultParameterButton_17.isChecked():
            self.ourinformation["level"] = 0.001      
        elif self.DefaultParameterButton_18.isChecked():
            self.ourinformation["level"] = 0.04
        elif self.DefaultParameterButton_19.isChecked():
            self.ourinformation["level"] = 0.10                          
    # output options

        if self.CoordinateButton.isChecked():
            self.ourinformation["Map"] = "Coordinate"      
        elif self.KmButton.isChecked():
            self.ourinformation["Map"] = "km" 

        if self.ContourButton.isChecked():
            self.ourinformation["contour"] = "contour"   
        elif self.NocontourButton.isChecked():
            self.ourinformation["contour"] = "nocontour"

        if self.FielddataButton.isChecked():
            self.ourinformation["Plot"] = "field" 
        elif self.NoFielddataButton.isChecked():
            self.ourinformation["Plot"] = "nofield"  

        if self.RunButton.isChecked():
            self.ourinformation["RunClick"] = "Run"
        if self.RecalcButton.isChecked():
            self.ourinformation["RecalcClick"] = "Recalculate" 
 
        print self.ourinformation

#__________________________________________________________________________________________________________________________
    # Methods for the oil of concern:    

    @pyqtSignature("")
    def on_submerged_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        self.OSCARButton.setVisible(True)
        self.OtherButton.setVisible(True)
        self.NodateButton.setVisible(True)
        self.GNOMEButton.setVisible(True)
        self.NodataButton.setVisible(False)
        self.UTMButton.setVisible(False)
        self.DecimalButton.setVisible(False)
        self.SampCampaignLabel_2.setText("Hydrodynamic Model Data")
        self.UploadSubmergedHydroButton.setText("Hydrodynamic Model Upload")
        areaselection = QMessageBox.question(self, "SOSim - Model Oil Type Simulator",
                                 "Please confirm that you want to model the submerged oil",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection

        if areaselection == QMessageBox.Yes:
        	print "This is submerged oil simulation" 
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Model Oil Type Simulator", "Please choose what kind of oil for simulation.")
        

    @pyqtSignature("")
    def on_sunken_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        self.OSCARButton.setVisible(False)
        self.OtherButton.setVisible(False)
        self.NodateButton.setVisible(False)
        self.GNOMEButton.setVisible(False)
        self.NodataButton.setVisible(True)
        self.UTMButton.setVisible(True)
        self.DecimalButton.setVisible(True)
        self.SampCampaignLabel_2.setText("Bathymetric Data")
        self.UploadSubmergedHydroButton.setText("Bathymetry File Upload")
        areaselection = QMessageBox.question(self, "SOSim - Model Oil Type Simulator",
                                 "Please confirm that you want to model the sunken oil",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "This is sunken oil simulation" 
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Model Oil Type Simulator", "Please choose what kind of oil for simulation.")

    # Methods for the Spill Location Layout:

    @pyqtSignature("")
    def on_RiverSpill_clicked(self):
        print "This is a River Spill"
        self.ourinformation["SpillPlace"] = "River"

    @pyqtSignature("")
    def on_OceanSpill_clicked(self):
        print "This is an Ocean Spill"
        self.ourinformation["SpillPlace"] = "Ocean"
        self.ourinformation['vxmin'] = self.vxMinSpinBox.text()
#__________________________________________________________________________________________________________________________
    # Methods for the type of spill:    

    @pyqtSignature("")
    def on_well_blowout_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Type of Spill",
                                 "Please confirm that this is a well-blowout spill",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:            
            print "This is well blowout oil simulation" 
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Set Spill Type", "Please select spill type.")     

    @pyqtSignature("")
    def on_vesselrelease_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Type of Spill",
                                 "Please confirm that this is a vessel release spill",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
            print "This is vessel release simulation" 
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Set Spill Type", "Please select spill type.")
#__________________________________________________________________________________________________________________________
    # Methods for the uncertainty estimates:    
    @pyqtSignature("")
    def on_DefaultParameterButton_6_clicked(self):
        print "This model for Best Guess"
        self.ourinformation["Method"] = "BestGuess"      

    @pyqtSignature("")
    def on_DefaultParameterButton_7_clicked(self):
        print "This model for Confidence Bounds"
        self.ourinformation["Method"] = "Minimum"
        level = self.vyMinSpinBox_3.text()
        self.ourinformation["confidence"] = level  
        QMessageBox.information(self, "SOSim - Confidence Level",
                                 "Make sure the Level of Confidence is from 0 to 1 (i.e. input 0.95 for 95% confidence level).")      

    @pyqtSignature("")
    def on_DefaultParameterButton_17_clicked(self):
        print "The confidence level is 0.1%"
        self.ourinformation["level"] = 0.001      

    @pyqtSignature("")
    def on_DefaultParameterButton_18_clicked(self):
        print "This model for Confidence Bounds"
        self.ourinformation["level"] = 0.04

    @pyqtSignature("")
    def on_DefaultParameterButton_19_clicked(self):
        print "This model for Confidence Bounds"
        self.ourinformation["level"] = 0.1                         
# output options
    @pyqtSignature("")
    def on_CoordinateButton_clicked(self):
        print "Coordinate scale"
        self.ourinformation["Map"] = "Coordinate"      

    @pyqtSignature("")
    def on_KmButton_clicked(self):
        print "km scale"
        self.ourinformation["Map"] = "km" 

    @pyqtSignature("")
    def on_ContourButton_clicked(self):
        print "Model output with contour"
        self.ourinformation["contour"] = "contour"   

    @pyqtSignature("")
    def on_NocontourButton_clicked(self):
        print "Model output without contour"
        self.ourinformation["contour"] = "nocontour"


    @pyqtSignature("")
    def on_FielddataButton_clicked(self):
        print "Plot with field"
        self.ourinformation["Plot"] = "field" 

    @pyqtSignature("")
    def on_NoFielddataButton_clicked(self):
        print "Plot without field"
        self.ourinformation["Plot"] = "field"   
#___________________start of ratio_________________________________________________________________________________________

#__________________end of ratio_____________________________________________________________________________________________        
    # Methods for the type of release:    

    @pyqtSignature("")
    def on_instantaneous_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Type of Release Selection",
                                 "Please confirm that the spill is an instantaneous spill",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "This is an instantaneous spill"

        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Type of Release Selection", "Please select the type of release.")

        self.ourinformation['instantaneous'] = 1
        self.ourinformation['Type'] = 'instantaneous'

        

    @pyqtSignature("")
    def on_continuous_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Type of Release Selection",
                                 "Please confirm that the spill is a continuous spill",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
            print "This is a continuous spill"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Type of Release Selection", "Please select oil release type.")
        self.ourinformation['Type'] = 'continuous'
   

#__________________________________________________________________________________________________________________________
    # Methods for the type of Hydrodynamic campaign:    

    @pyqtSignature("")
    def on_OSCARButton_clicked(self):
        self.textEditor1.setVisible(True)
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Hydrodynamic Model Selection",
                                 "Please confirm that you have chosen to upload OSCAR model output",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "OSCAR model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Hydrodynamic Model Selection", "Please choose the hydrodynamic model")
        

    @pyqtSignature("")
    def on_GNOMEButton_clicked(self):
        self.textEditor1.setVisible(True)
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Hydrodynamic Model Selection",
                                 "Please confirm that you have chosen to upload GNOME model output",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "GNOME model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Hydrodynamic Model Selection", "Please choose the hydrodynamic model")

   

    @pyqtSignature("")
    def on_OtherButton_clicked(self):
        self.textEditor1.setVisible(True)
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Hydrodynamic Model Selection",
                                 "Please confirm that the other model has the required format",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "Other model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Hydrodynamic Model Selection", "Please choose the hydrodynamic model")
        
    @pyqtSignature("")
    def on_NodateButton_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        self.textEditor1.setVisible(False)
        areaselection = QMessageBox.question(self, "SOSim - Hydrodynamic Model Selection",
                                 "Please confirm that no model has been selected",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	self.UploadSubmergedHydroButton.setEnabled(False)                             
        	print "Submerge No model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Hydrodynamic Model Selection", "Please choose the hydrodynamic model")

########################################################
    @pyqtSignature("")
    def on_DecimalButton_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Bathymetry Upload",
                                 "Please confirm that you have chosen to upload Decimal degree coordinates",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "GNOME model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Bathymetry Upload", "Please choose the bathymetry upload type")

   

    @pyqtSignature("")
    def on_UTMButton_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Bathymetry Upload",
                                 "Please confirm you choose to upload UTM coordinates",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	print "Other model has been selected"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Bathymetry Upload", "Please choose the bathymetry upload type")
        
    @pyqtSignature("")
    def on_NodataButton_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Bathymetry Upload",
                                 "Please confirm that there is no bathymetric upload. SOSim will use a default module to extract bathymetric data for simulation.",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
        	self.UploadSubmergedHydroButton.setEnabled(False)                             
        	print "No bathymetry upload"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Bathymetry Upload", "Please upload bathymetry file")   


#__________________________________________________________________________________________________________________________
    # Methods for the Parameter Range:    

    @pyqtSignature("")
    def on_DefaultParameterButton_clicked(self):
        """Sets x_nodes and y_nodes global variables to the default set by user in the options menu, or 25 if unchanged"""
        areaselection = QMessageBox.question(self, "SOSim - Parameter Setting",
                                 "Please confirm that you selected the parameter default",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
	        if self.submerged.isChecked():
	        	self.ourinformation['vxmin'] = -3.0
	        	self.ourinformation['vxmax'] = 3.0
	        	self.ourinformation['vymin'] = -3.0
	        	self.ourinformation['vymax'] = 3.0
	        	self.ourinformation['dxmin'] = 0.01
	        	self.ourinformation['dxmax'] = 0.89 
	        	self.ourinformation['dymin'] = 0.01
	        	self.ourinformation['dymax'] = 0.89 
	        if self.sunken.isChecked():
	        	if self.RiverSpill.isChecked():
	        		self.ourinformation['vxmin'] = -0.5#-2.4
	        		self.ourinformation['vxmax'] = 0.5#2.4
	        		self.ourinformation['vymin'] = -67.1
	        		self.ourinformation['vymax'] = 67.1
	        		self.ourinformation['dxmin'] = 0.00001
	        		self.ourinformation['dxmax'] = 0.0001 
	        		self.ourinformation['dymin'] = 0.01
	        		self.ourinformation['dymax'] = 0.6 
	        	if self.OceanSpill.isChecked():
	        		self.ourinformation['vxmin'] = -3.0
	        		self.ourinformation['vxmax'] = 3.0
	        		self.ourinformation['vymin'] = -3.0
	        		self.ourinformation['vymax'] = 3.0
	        		self.ourinformation['dxmin'] = 0.01
	        		self.ourinformation['dxmax'] = 0.89 
	        		self.ourinformation['dymin'] = 0.01
	        		self.ourinformation['dymax'] = 0.89 
	        print self.ourinformation
	        print "__________________________________________________________"
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Parameter Setting", "Please set the parameter range")

    @pyqtSignature("")
    def on_UserDefineParameterButton_clicked(self):
        """Gets x_nodes and y_nodes input by the user and generates global variable value"""
        areaselection = QMessageBox.question(self, "SOSim - Parameter Setting",
                                 "Please confirm that you selected the parameter define",
                                 QMessageBox.Yes|QMessageBox.No)
        areaselection
        if areaselection == QMessageBox.Yes:
           print "user setting"
        
        if areaselection == QMessageBox.No:
            QMessageBox.information(self, "SOSim - Parameter Setting", "Please set the parameter range")
        

#__________________________________________________________________________________________________________________________
    # Methods for the Calculation Times Layout:    

    @pyqtSignature("QTime")
    def on_CalcHourEdit_timeChanged(self):
        calchour = self.CalcHourEdit.time()
        hours = calchour.hour()
        minutes = calchour.minute()
        HourCalc = hours + (minutes/60.0)   
        HourCalc = float(HourCalc)
        print "HourCalc:", HourCalc
        return HourCalc

        
    @pyqtSignature("QDate")
    def on_CalcDateEdit_dateChanged(self):
        calcdate = self.CalcDateEdit.date()
        months = calcdate.month()
        days = calcdate.day()
        years = calcdate.year()
        DateCalc = [years, months, days]
        print "DateCalc:", DateCalc
        return DateCalc

  
    def SetCalcTime(self):
        HourCalc = self.on_CalcHourEdit_timeChanged()
        DateCalc = self.on_CalcDateEdit_dateChanged()
        DateCalc.append(HourCalc)
        TimeCalc = numpy.array(DateCalc)
        TimeZero = numpy.array(self.SetTimeZero())
        TClist = TimeCalc - TimeZero
        daysinmonth = (calendar.monthrange(int(TimeZero[0]),int(TimeZero[1])))[1] # accounts for leap years.
        print "DAYS IN MONTH are %s" % daysinmonth
        a = 1
        daysinyear = 0
        while a <= 12:
            daysinyear = daysinyear + (calendar.monthrange(int(TimeZero[0]),int(a)))[1]
            a += 1
        if TClist[3] < 0.0:
            TClist[2] = TClist[2]-1.0
            TClist[3] = 24.0 + TClist[3]
        if TClist[2] < 0.0:
            TClist[1] = TClist[1]-1.0
            TClist[2] = daysinmonth + TClist[2]
        if TClist[1] < 0.0:
            TClist[0] = TClist[0]-1.0
            TClist[1] = 12.0 + TClist[1]
        if TClist[0] < 0.0:
            QMessageBox.critical(self, "SOSim - Dates error", "Prediction dates must follow a spill event.")
            self.CalcDateEdit.selectAll()
            self.CalcDateEdit.setFocus()
        else:
            TC = (daysinyear * TClist[0]) + (daysinmonth * TClist[1]) + TClist[2] + (TClist[3]/24.0)
            if TC > daysinyear:
                QMessageBox.warning(self, "SOSim - Dates error", "Revise spill and prediction years. If correct, the prediction date might be invalid.")
            if TC > 90:
#            if TC > daysinmonth:
                QMessageBox.warning(self, "SOSim - Long Term Date", "The prediction time is considered long term by SOSim. Check that sampling campaigns which dates are close to the requested prediction time are loaded to the model for likelihood updates and to avoid inaccuracy")
        
        return TC


    @pyqtSignature("")    
    def on_AddCalcTimeButton_clicked(self):
        """Add the current prediction time to the registry"""

        # add predict time
        calchour = self.CalcHourEdit.time()
        s1 = calchour.toString(Qt.TextDate)
        calcdate = self.CalcDateEdit.date()
        s2 = calcdate.toString("yyyy-MM-dd")
        self.ourinformation['OurTime'].append(s2 + " " + s1)
        print self.ourinformation['OurTime']
        print self.CalcTimeNumberSpinBox.value()

        self.CalcTimeNumberSpinBox.setValue(len(self.ourinformation['OurTime']))
        
        TC = self.SetCalcTime()
        
        if self.st != []:
            B = [max(self.DLcon[vld]) for vld in xrange(len(self.st))]
            hiIndex = B.index(max(B))
            latestST = max(self.st)
            self.admST = self.st[hiIndex] + self.retardation
        else:
            K = open(PlayPath + "/my_MaxST.txt", "r") #Saved as calibration and then saved to path2 to be readily available to "open" command.
            allST = pickle.load(K)
            K.close()
            L = open(PlayPath + "/my_DLcon.txt", "r") #Saved as calibration and then saved to path2 to be readily available to "open" command.
            allDLcon = pickle.load(L)
            L.close()
            B = [max(allDLcon[vld]) for vld in xrange(len(allST))]
            hiIndex = B.index(max(B))
            latestST = max(allST)
            self.admST = allST[hiIndex] + self.retardation
        self.openOcean=True
        if self.openOcean is False: # and if boundaries selected from map: (add this condition)
            print "false openecean"
            coastTime = self.stopTime()
            print "coastTime: %s" % coastTime

            if TC < coastTime:
                
                if TC != 0.0 and TC >= latestST + self.retardation:
                    TC = TC - admST ###
                    self.t.append(TC)
                    print "t:", self.t
                    QMessageBox.information(self, "SOSim - Prediction Times", "You have added one date for prediction.")
                else:
                    QMessageBox.warning(self, "SOSim - Not Valid Date", "Prediction dates must be different from the spill date and go on or after the latest sampling campaign time.")

            if TC >= coastTime:
                stop = QMessageBox.question(self, "SOSim - Not Defined Prediction",
                                     "The calculation date you are trying to add is of a time close to the time projected for the sunken oil to reach the coast. SOSim will not accurately predict oil's behavior passed an accumulation point. Do you want to keep adding this prediction time anyhow?",
                                     QMessageBox.Yes|QMessageBox.No)
                stop
                if stop == QMessageBox.Yes:
                    if TC != 0.0 and TC >= latestST + self.retardation:
                        TC = TC - self.admST ###
                        self.t.append(TC)
                        print "t:", self.t
                        QMessageBox.information(self, "SOSim - Prediction Times", "You have added one date for prediction.")
                    else:
                        QMessageBox.warning(self, "SOSim - Not Valid Date", "Prediction dates must be different from the spill date and go on or after the latest sampling campaign time.")
        
                if stop == QMessageBox.No:
                    pass
                
        if self.openOcean is True: 
            
            if TC != 0.0 and TC >= latestST + self.retardation:
                TC = TC - self.admST ###
                self.t.append(TC)
                print "t:", self.t
                QMessageBox.information(self, "SOSim - Prediction Times", "You have added one date for prediction.")
            else:
                QMessageBox.warning(self, "SOSim - Not Valid Date", "Prediction dates must be different from the spill date and go on or after the latest sampling campaign time.")
                
        print "t:", self.t
        enable = len(self.t) > 1
        
        self.morning = [self.t[0] + self.admST, 0] 
        self.evening = [self.t[len(self.t)-1] + self.admST, len(self.t)]
        print "enable"
            
        

    @pyqtSignature("")
    def on_RemoveCalcTimeButton_clicked(self):
        
        if len(self.ourinformation['OurTime']) == 0:
            QMessageBox.information(self,"SOSim - Prediction Times", "You have zero prediction date.")
        else:
            (self.ourinformation['OurTime']).pop()
            print self.ourinformation['OurTime']

            self.CalcTimeNumberSpinBox.setValue(len(self.ourinformation['OurTime']))            

            QMessageBox.information(self, "SOSim - Prediction Times", "You have deleted one prediction date.")
        

        
#__________________________________________________________________________________________________________________________
    # Methods for the Run Layout:


    def checkMissingInfo(self):
        if (self.st == []) and (self.importedCalibration != True):
            QMessageBox.warning(self, "SOSim - Missing Input", "Please upload sampling campaign(s) or import an existing calibration file.")
            
                

    @pyqtSignature("")
    def on_CalibrateButton_clicked(self):
        T1 = time.asctime()
        self.progressBar.reset()

        Dxmin = self.popDialog.DxMinSpinBox.value()
        Dymin = self.popDialog.DyMinSpinBox.value()
        Dxmax = self.popDialog.DxMaxSpinBox.value()
        Dymax = self.popDialog.DyMaxSpinBox.value()
        vxmin = self.popDialog.vxMinSpinBox.value()
        vymin = self.popDialog.vyMinSpinBox.value()
        vxmax = self.popDialog.vxMaxSpinBox.value()
        vymax = self.popDialog.vyMaxSpinBox.value()
        romin = self.popDialog.roMinSpinBox.value()
        romax = self.popDialog.roMaxSpinBox.value()
        print "ranges:", [Dxmin, Dymin, Dxmax, Dymax, vxmin, vymin, vxmax, vymax, romin, romax]

        self.progressBar.setValue(10)
        InsPrelim = SOSimCore.Preliminars()
        args = InsPrelim.doAll(Dxmin, Dymin, Dxmax, Dymax, vxmin, vymin, vxmax, vymax, romin, romax)
        
        delta = args[0]
        newsze = args[1]
        u = args[2]
        vx = args[3]
        vy = args[4]
        Dx = args[5]
        Dy = args[6]
        ro = args[7]
        g = args[8]

        valid = InsPrelim.methodValid()
        GammaPossible = InsPrelim.methodGammaPossible()
        self.progressBar.setValue(20)

        if len(self.st) == 1:
            maxN = len(self.DLcon[0])
        if len(self.st) > 1:
            maxN = 0
            k = 0
            while k < (len(self.DLcon))-1:
                maxN = max(maxN, max(len(self.DLcon[k]), len(self.DLcon[k+1])))
                k += 1
        print "maxN: %s" % maxN
            
        self.x0 = self.set_x0()
        self.y0 = self.set_y0()
        self.zone0 = self.set_zone0()
        
        print "Calculations of the Likelihood Function started %s" % time.asctime()
        self.progressBar.setValue(30)
        
        argus = args
        argus.append(self.xclicks)
        argus.append(self.yclicks)
        
        InsLFClass = SOSimCore.LF(self.openOcean, self.boundSegments, GammaPossible, valid, maxN, self.DLx, self.DLy, self.DLcon,\
                                  self.x0, self.y0, self.sx0, self.sy0, self.st, *argus)

        print "LFClass instance created %s" % time.asctime()
        self.progressBar.setValue(80)

        self.x0y0DueSinkingRetardation()

        InsLFClass.calculateLV(maxN, self.retardation, self.x0, self.y0, self.openOcean)
        
        self.progressBar.setValue(90)

        self.TitleEdit.setText("Prediction for Spill %s at Time %.3f Days" %(self.spillName(), self.t[0]+ self.admST))
        
        # To pass LikelihoodFunction to Recalculate method:
        self.valid = valid
        self.args = args
        self.GammaPossible = GammaPossible

        self.RecalcButton.setEnabled(True)

        self.progressBar.setValue(100)


    @pyqtSignature("")
    def on_RunButton_clicked(self):

        global myfile_list
        global myfile1_list
        global myfile2_list        
        global cur
        global k_zt
        self.Getinformation()
        self.ourinformation["Run"] = "Run"

        if self.submerged.isChecked():
            # myfile,myfile1 = submerged.submerged_main(self.ourinformation,self.progressBar)
            for i in range(len(self.ourinformation['OurTime'])):
                self.progressBar.setValue(0)
                self.ourinformation['PredictTime'] = self.ourinformation['OurTime'][i]
                myfile,myfile1,myfile2 = submerged.submerged_main(self.ourinformation,self.progressBar)
                myfile_list.append(myfile)
                myfile1_list.append(myfile1)
                myfile2_list.append(myfile2)                

        elif  self.sunken.isChecked():
            for i in range(len(self.ourinformation['OurTime'])):
                self.progressBar.setValue(0)
                print self.ourinformation['OurTime']                
                self.ourinformation['PredictTime'] = self.ourinformation['OurTime'][i]
                myfile,myfile1,myfile2 = sunken.sunken_main(self.ourinformation,self.progressBar)
                myfile_list.append(myfile)
                myfile1_list.append(myfile1)
                myfile2_list.append(myfile2)  

        print "SOSim max min",self.x_min,self.x_max,self.y_min,self.y_max

        self.progressBar.setValue(90)

        self.progressBar.setValue(100)
        cur = len(myfile_list)-1
        k_zt = cur
        print 'k_zt',k_zt
        self.addRasterImage1()
                
        myfile1 = myfile1_list[cur]
        for i in  range(len(myfile1)-1):
            if(myfile1[i]=='/' and myfile1[i+1]!='/'):
                number = i
        self.MyLegendLoad(myfile1,myfile1[i+1:])

        myfile2 = myfile2_list[cur]
        for i in  range(len(myfile2)-1):
            if(myfile2[i]=='/' and myfile2[i+1]!='/'):
                number = i
        self.MyLegendLoad2(myfile2,myfile2[i+1:])    
        print cur

    @pyqtSignature("")
    def on_RecalcButton_clicked(self):
        """Recalculate model for new area but keeping the current campaigns and therefore likelihood function"""

        global myfile_list
        global myfile1_list
        global myfile2_list        
        global cur
        global k_zt
        self.Getinformation()

        self.ourinformation["Run"] = "Recalc"
        #T1 = time.asctime()
        #self.TitleEdit.setText("Model Recalculation Set Up for Spill %s" % self.spillName())
        #self.progressBar.setValue(10)
        #self.layers.pop(-1)
        #self.progressBar.setValue(25)
        diffBounds = QMessageBox.question(self, "SOSim - About New Boundary Conditions",
                                 "Do you want to keep the same number of nodes and boundaries selected for the previous area?",
                                 QMessageBox.Yes|QMessageBox.No)
        diffBounds
        if diffBounds == QMessageBox.Yes:
            pass
        if diffBounds == QMessageBox.No:
            argus.pop(-1) # deleting previous boundaries to add the new ones
            argus.pop(-1)
            QMessageBox.information(self, "SOSim - Set Boundaries", "Please click on 'No nearby boundary' or 'Select Boundary' buttons")
            argus.append(self.xclicks)
            argus.append(self.yclicks)
    
        spillname = self.spillName()

        if self.submerged.isChecked():
            # myfile,myfile1 = submerged.submerged_main(self.ourinformation,self.progressBar)
            for i in range(len(self.ourinformation['OurTime'])):
                self.progressBar.setValue(0)
                self.ourinformation['PredictTime'] = self.ourinformation['OurTime'][i]
                myfile,myfile1,myfile2 = submerged.submerged_main(self.ourinformation,self.progressBar)
                myfile_list.append(myfile)
                myfile1_list.append(myfile1)
                myfile2_list.append(myfile2)                
                

        elif  self.sunken.isChecked():
            for i in range(len(self.ourinformation['OurTime'])):
                self.progressBar.setValue(0)
                print self.ourinformation['OurTime']                
                self.ourinformation['PredictTime'] = self.ourinformation['OurTime'][i]
                myfile,myfile1,myfile2 = sunken.sunken_main(self.ourinformation,self.progressBar)
                myfile_list.append(myfile)
                myfile1_list.append(myfile1)
                myfile2_list.append(myfile2)  

        print "SOSim max min",self.x_min,self.x_max,self.y_min,self.y_max
               
 
        self.progressBar.setValue(90)

        self.progressBar.setValue(100)
        cur = len(myfile_list)-1
        k_zt = cur
        print 'k_zt',k_zt
        self.addRasterImage1()
                
        myfile1 = myfile1_list[cur]
        for i in  range(len(myfile1)-1):
            if(myfile1[i]=='/' and myfile1[i+1]!='/'):
                number = i
        self.MyLegendLoad(myfile1,myfile1[i+1:])

        myfile2 = myfile2_list[cur]
        for i in  range(len(myfile2)-1):
            if(myfile2[i]=='/' and myfile2[i+1]!='/'):
                number = i
        self.MyLegendLoad2(myfile2,myfile2[i+1:])    
        print cur

        print "Recalculation is done"
        # self.progressBar.setValue(40)
        
        
#__________________________________________________________________________________________________________________________
    # Methods for the ViewTime Layout:

    def rearangeLayers(self):
        self.layers.pop(-1)
        self.MyWorldLayer(ext = False)
        

    def nextTimeGenerator(self):
        class NoMoreCalcTimes(Exception): pass
        try:
            i = self.morning[1]
            while i < len(self.t):
                yield [self.t[i]+ self.admST, i]
                i += 1
        except NoMoreCalcTimes, e:
            QMessageBox.warning(self, "There are no further prediction times to show", unicode(e))
            self.NextTimeButton.setEnabled()


    @pyqtSignature("")       
    def on_NextTimeButton_clicked(self):

        global k_zt
        global cur
        length = len(myfile_list)
        cur  = (cur+1)%length
        print "cur:",cur
        k_zt = cur
        self.addRasterImage1()
        myfile1 = myfile1_list[cur]
        for i in  range(len(myfile1)-1):
            if(myfile1[i]=='/' and myfile1[i+1]!='/'):
                    number = i
            self.MyLegendLoad(myfile1,myfile1[i+1:])


    def prevTimeGenerator(self):
        class NoMoreCalcTimes(Exception): pass
        try:
            i = self.evening[1]
            while i >= 0:
                yield [self.t[i]+ self.admST, i]
                i -= 1
        except NoMoreCalcTimes, e:
            QMessageBox.warning(self, "SOSim - There are no previous prediction times to show", unicode(e))
            self.NextTimeButton.setEnabled()
                                   

    @pyqtSignature("")       
    def on_PreviousTimeButton_clicked(self):
        global k_zt
        global cur
        length = len(myfile_list)
        cur  = (cur-1)%length
        print "cur:",cur
        k_zt = cur
        self.addRasterImage1()
        myfile1 = myfile1_list[cur]
        for i in  range(len(myfile1)-1):
            if(myfile1[i]=='/' and myfile1[i+1]!='/'):
                    number = i
            self.MyLegendLoad(myfile1,myfile1[i+1:])

        
#__________________________________________________________________________________________________________________________
    # Methods for the Pan Layout:

    def moveWorldNSEW(self, disable):
        """Create a rectangle to cover the new extent when panning with NSEW options"""
        rect = QgsRectangle(self.x_min, self.y_min, self.x_max, self.y_max) 
        self.canvas.setExtent(rect)
        self.canvas.refresh()
        # Disable not convenient buttons:
        if self.PanE.isChecked() or self.PanW.isChecked() or self.PanN.isChecked() or self.PanS.isChecked() or self.PanC.isChecked() or self.RunButton.isEnabled():
            self.RunButton.setEnabled(disable)
            #self.AutoSelectAreaButton.setEnabled(disable)
            #self.ManSelectAreaButton.setEnabled(disable)
            if self.DefaultNodesRadioButton.isChecked():
                self.DefaultNodesRadioButton.setChecked(disable)
            if self.UserDefinedNodesRadioButton.isChecked():
                self.UserDefinedNodesRadioButton.setChecked(disable)


    @pyqtSignature("")
    def on_PanE_clicked(self):
        """MoveAreaEast for recalculation"""
        X = self.x_min
        XX = self.x_max
        self.x_min = XX
        self.x_max = XX + (XX-X)
        disable = self.PanE.isChecked()
        self.moveWorldNSEW(disable)
        #QMessageBox.information(self, "SOSim - Model Recalculation", "If you are done panning, please proceed selecting number of modeling nodes and land boundary options, then click on 'Recalculate'.")


    @pyqtSignature("")
    def on_PanW_clicked(self):
        """MoveAreaWest for recalculation"""
        X = self.x_min
        XX = self.x_max
        self.x_min = X - (XX-X)
        self.x_max = X
        disable = self.PanW.isChecked()
        self.moveWorldNSEW(disable)
        #QMessageBox.information(self, "SOSim - Model Recalculation", "If you are done panning, please proceed selecting number of modeling nodes and land boundary options, then click on 'Recalculate'.")

            
    @pyqtSignature("")
    def on_PanN_clicked(self):
        """MoveAreaNorth for recalculation"""
        Y = self.y_min
        YY = self.y_max
        self.y_min = YY
        self.y_max = YY + (YY-Y)
        disable = self.PanN.isChecked()
        self.moveWorldNSEW(disable)
        #QMessageBox.information(self, "SOSim - Model Recalculation", "If you are done panning, please proceed selecting number of modeling nodes and land boundary options, then click on 'Recalculate'.")
            
    @pyqtSignature("")    
    def on_PanS_clicked(self):
        """MoveAreaSouth for recalculation"""
        Y = self.y_min
        YY = self.y_max
        self.y_min = Y - (YY-Y)
        self.y_max = Y
        disable = self.PanS.isChecked()
        print disable
        self.moveWorldNSEW(disable)
        #QMessageBox.information(self, "SOSim - Model Recalculation", "If you are done panning, please proceed selecting number of modeling nodes and land boundary options, then click on 'Recalculate'.")
    
    @pyqtSignature("")
    def on_PanC_clicked(self):
        """MoveAreaEast for recalculation"""      
        disable = self.PanC.isChecked()
        print disable
        self.showSpillSiteCanvas(0.15)
 #__________________________________________________________________________________________________________________________
    # Methods for the Cancel Layout:
    

    @pyqtSignature("") 
    def on_CancelButton_clicked(self):
        sys.exit()
        
#__________________________________________________________________________________________________________________________
# Rubberband class for interactive polyline drafting on canvas:


class DrawPolylineMapTool( QgsMapTool ):
    """Tool to draw a polyline on the map canvas"""
    def __init__( self, parent, canvas, boundVertices):
        QgsMapTool.__init__( self, canvas)
        self.canvas = canvas
        self.boundVertices = boundVertices 

        self.mXCoords = []
        self.mYCoords = []

        self.mRubberBand = QgsRubberBand( self.canvas, False )
        self.mRubberBand.setColor( QColor( 255,255,48,100 ) )
        self.mRubberBand.setWidth( 2 )

        self.bTerminada = False

        self.connect( parent, SIGNAL( "delete_polyline" ), self.reset )
        

    def canvasPressEvent(self, event):
        if self.bTerminada: 
            self.bTerminada = False
            self.reset()


    def canvasMoveEvent(self, event):
        if not self.bTerminada:
            point = self.toMapCoordinates(QPoint(event.pos().x(), event.pos().y()))
            self.mRubberBand.movePoint(point, 0)


    def canvasReleaseEvent(self, event):
        if not self.bTerminada:
            point = self.toMapCoordinates(QPoint(event.pos().x(), event.pos().y()))
            if event.button() == 1: # left button
                self.addPoint(point)
            elif event.button() == 2: # right button
                self.bTerminada = True
                self.reset()


    def addPoint(self, point):
        if len(self.mXCoords) > 0 and point.x() == self.mXCoords[0] and point.y() == self.mYCoords[0]:
            return

        self.mXCoords.append(point.x())
        self.mYCoords.append(point.y())

        self.mRubberBand.addPoint( point, True, 0 )

        if len(self.mXCoords) == self.boundVertices:
            self.bTerminada = True
            self.emit(SIGNAL("polyline_finished"), [self.mXCoords, self.mYCoords])
            

    def deactivate(self):
        self.emit(SIGNAL("toolbar_deactivated"))
        

    def reset(self):
        self.mXCoords = []
        self.mYCoords = []
        self.mRubberBand.reset(False)


    def isZoomTool(self):
        return False


#__________________________________________________________________________________________________________________________
# Dialog class for interactive options:

class myDialog(QDialog, Ui_MyDialog):
    def __init__(self, parent = None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

        self.DxMinSpinBox.setValue(0.01)
        self.DyMinSpinBox.setValue(0.01)
        self.DxMaxSpinBox.setValue(0.89)
        self.DyMaxSpinBox.setValue(0.89)
        self.vxMinSpinBox.setValue(-3.00)
        self.vyMinSpinBox.setValue(-3.00)
        self.vxMaxSpinBox.setValue(3.00)
        self.vyMaxSpinBox.setValue(3.00)
        self.roMinSpinBox.setValue(-0.99)
        self.roMaxSpinBox.setValue(0.99)
        self.EWdefault.setValue(25)
        self.NSdefault.setValue(25)
        self.resolutionMax.setValue(55.56)


        self.Dxmin = 0.01
        self.Dymin = 0.01
        self.Dxmax = 0.89     # Units in km^2/day
        self.Dymax = 0.89
        self.vxmin = -3.0
        self.vymin = -3.0
        self.vxmax = 3.
        self.vymax = 3.
        self.romin = -0.99
        self.romax = 0.99
        self.user_nodes_x = 25
        self.user_nodes_y = 25
        self.defaultResolution = 55.56*0.00020698344132469402447804175665947

    
    @pyqtSignature("")
    def on_ApplyButton_clicked(self):
        
        self.Dxmin = self.DxMinSpinBox.value()
        self.Dymin = self.DyMinSpinBox.value()
        self.Dxmax = self.DxMaxSpinBox.value()
        self.Dymax = self.DyMaxSpinBox.value()
        self.vxmin = self.vxMinSpinBox.value()
        self.vymin = self.vyMinSpinBox.value()
        self.vxmax = self.vxMaxSpinBox.value()
        self.vymax = self.vyMaxSpinBox.value()
        self.romin = self.roMinSpinBox.value()
        self.romax = self.roMaxSpinBox.value()
        self.user_nodes_x = self.EWdefault.value()
        self.user_nodes_y = self.NSdefault.value()

        self.defaultResolution = self.resolutionMax.value()*0.00020698344132469402447804175665947

        print [self.Dxmin, self.Dymin, self.Dxmax, self.Dymax, self.vxmin, self.vymin, self.vxmax, self.vymax, self.romin,\
               self.romax, self.user_nodes_x, self.user_nodes_y, self.defaultResolution]
           
        self.close()
        

    def ranges(self):
        return [self.Dxmin, self.Dymin, self.Dxmax, self.Dymax, self.vxmin, self.vymin, self.vxmax, self.vymax, self.romin, self.romax]


    @pyqtSignature("")
    def on_RestoreButton_clicked(self):
        
        self.Dxmin = 0.01
        self.DxMinSpinBox.setValue(0.01)
        self.Dymin = 0.01
        self.DyMinSpinBox.setValue(0.01)
        self.Dxmax = 0.89     # Units in km^2/day
        self.DxMaxSpinBox.setValue(0.89)
        self.Dymax = 0.89
        self.DyMaxSpinBox.setValue(0.89)
        self.vxmin = -3.0
        self.vxMinSpinBox.setValue(-3.00)
        self.vymin = -3.0
        self.vyMinSpinBox.setValue(-3.00)
        self.vxmax = 3.
        self.vxMaxSpinBox.setValue(3.00)
        self.vymax = 3.
        self.vyMaxSpinBox.setValue(3.00)
        self.romin = -0.999
        self.roMinSpinBox.setValue(-0.99)
        self.romax = 0.999
        self.roMaxSpinBox.setValue(0.99)
        self.user_nodes_x = 25
        self.EWdefault.setValue(25)
        self.user_nodes_y = 25
        self.NSdefault.setValue(25)
        self.defaultResolution = 55.56*0.00020698344132469402447804175665947
        self.resolutionMax.setValue(55.56)


    @pyqtSignature("")
    def on_CancelButton_clicked(self):
        self.DxMinSpinBox.setValue(self.Dxmin)
        self.DyMinSpinBox.setValue(self.Dymin)
        self.DxMaxSpinBox.setValue(self.Dxmax)
        self.DyMaxSpinBox.setValue(self.Dymax)
        self.vxMinSpinBox.setValue(self.vxmin)
        self.vyMinSpinBox.setValue(self.vymin)
        self.vxMaxSpinBox.setValue(self.vxmax)
        self.vyMaxSpinBox.setValue(self.vymax)
        self.roMinSpinBox.setValue(self.romin)
        self.roMaxSpinBox.setValue(self.romax)
        self.EWdefault.setValue(self.user_nodes_x)
        self.NSdefault.setValue(self.user_nodes_y)
        self.resolutionMax.setValue(self.defaultResolution/0.00020698344132469402447804175665947)
        self.close()
   
def main():
    
    # initialize qgis libraries

    QgsApplication.setPrefixPath(qgis_prefix, True)
    app = QgsApplication([],True)
    app.initQgis()
    QCoreApplication.setLibraryPaths([])
    # Launch application
    # app = QApplication(sys.argv)
    app.setOrganizationName("NOAA-CRRC")
    app.setOrganizationDomain(" ")
    app.setApplicationName("SOSim2.0 - Subsurface Oil Simulator")
    form = SOSimMainWindow()
    form.show()
    app.exec_()
    

if __name__ == "__main__":
    main()
