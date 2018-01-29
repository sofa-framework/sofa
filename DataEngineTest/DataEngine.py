#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine(Sofa.PythonScriptDataEngine):


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
        



