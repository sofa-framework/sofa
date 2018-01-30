#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine(Sofa.PythonScriptDataEngine):    

    def parse(self):
	MyBoxROI = BoxROI(-30, -30, 8, 0, 30, 50, self.position)
        Tetras = self.tetrahedra	
        #MyBoxROI.calcTetrasInROI(Tetras)
	self.addNewOutput('tetrahedraInliers',datatype='t',value=Tetras)
	#self.addNewOutput('TetrahedraInliers',datatype='t')
	#self.addNewOutput('TetrahedraOutliers',datatype='t',value=MyBoxROI.Outliers)
	#self.TetrahedraInliers = MyBoxROI.Inliers
	#self.TetrahedraInliers = []
	#self.TetrahedraInliers = MyBoxROI.Inliers	
	#print self.TetrahedraInliers
	print 'end parsing ...'

    def update(self):
        print 'blup'        

    def init(self):
	print 'Begin init ...'

	print 'end init ...'



