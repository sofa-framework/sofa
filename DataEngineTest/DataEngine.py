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
        self.addNewInput('DataNameI',datatype='t', dataclass='sd',help='This input is used for blah')
        #self.addNewOutput('DataNameO',value=[1,12,4,5,24,3,4,5,3,4,5,3], dataclass='sd',help='This input is used for blah')
	
	
	root=self.getContext()
	MrT = root.getObject('container').findData('tetrahedra')
  	#print MrT
        #self.addNewOutput('MrT', 'Properties','help', 't', MrT)

        #self.addNewInput(self.root.findData(''))

	#self.addNewData('test3','Properties','help','s','asd');
        #self.addField('Input')
        
        #self.init()
	
	#print self.tetrahedraOutlierss
	#MyBoxROI.isTetraInBoxROI(Tetras[1])



