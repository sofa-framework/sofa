#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa




class BoxROI:

    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, PointCoords):
    	self.xmin = xmin
    	self.xmax = xmax
    	self.ymin = ymin
    	self.ymax = ymax
	self.zmin = zmin
	self.zmax = zmax
	self.PointCoords = PointCoords

    def isTetraInBoxROI(self, Tetra):
	#print Tetra[0]

	c1 = self.PointCoords[Tetra[0]]
	c2 = self.PointCoords[Tetra[1]]
	c3 = self.PointCoords[Tetra[2]]
	c4 = self.PointCoords[Tetra[2]]
	CenterX = (c1[0]+c2[0]+c3[0]+c4[0])/4
	CenterY = (c1[1]+c2[1]+c3[1]+c4[1])/4
	CenterZ = (c1[2]+c2[2]+c3[2]+c4[2])/4
	if CenterX > self.xmin and CenterX < self.xmax:
		if CenterY > self.ymin and CenterY < self.ymax:
			if CenterZ > self.zmin and CenterZ < self.zmax:
				return True
	return False

    def calcTetrasInROI(self, Tetras):
	NTetras = len(Tetras)
	self.Inliers = []
	self.Outliers = []
	for i in range(0,NTetras):
		if self.isTetraInBoxROI(Tetras[i]):
			self.Inliers.append(Tetras[i])
		else:
			self.Outliers.append(Tetras[i])	

	

		

class MyDataEngine(Sofa.PythonScriptDataEngine):
#    def init(self):
#	self.addInput( self.createData("mydatafromscript", "This is help", "f", 1.0) )
#	self.addOutput( self.createData("mydatafromscript2", "This is help", "f", 1.0) )
#	self.addInput("toto", [12.0, 2.0])	    	


    def update(self):
        print 'blup'        
        #self.init()
	#print(''+str(self.name))	
	#self.addInput( self.createData("mydatafromscript", "This is help", "f", 1.0) )
	#print(self.tetrahedra)
	#print(self.position)
	#print(self.mydata)	
	#self.mydatafromscript2 = 2*self.mydatafromscript

    def init(self):
        MyBoxROI = BoxROI(-30, -30, 8, 0, 30, 50, self.position)
        Tetras = self.tetrahedra
        MyBoxROI.calcTetrasInROI(Tetras)
        self.tetrahedraOutliers = MyBoxROI.Outliers	
        self.tetrahedraComputed = MyBoxROI.Inliers	
        #print self.myblah
        #self.myblah = 40
        #print self.myblah
        #self.addNewOutput('teasting','Properties','help','t',[[1,12,4,5],[24,3,4,5],[3,4,5,3],[3,4,5,56]])
        #self.addNewInput('DataNameI', datatype='t', value='@container.tetrahedra',help='This input is used for blah')

        #self.addNewOutput('DataNameO',datatype='t',value=[1,12,4,5,24,3,4,5,3,4,5,3],help='This output is used for blah')
		
	#root=self.getContext()
	#MrT = root.getObject('container').findData('tetrahedra')
	MrTT= self.findData('tetrahedra').value
	#MrTT= Tetras
	for i in range(0,len(MrTT)):
		for j in range(0,len(MrTT[i])):			
			MrTT[i][j] = MrTT[i][j]+1

	self.addNewOutput('output1', datatype='t',value=MrTT)	

        



