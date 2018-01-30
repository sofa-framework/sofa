#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine(Sofa.PythonScriptDataEngine):

    def poarse(self):
	print 'poarsing 2'	

    def update(self):
        print 'blup'        

    def init(self):
        MyBoxROI = BoxROI(0, -30, 8, 30, 30, 50, self.position)
	self.addNewInput('TetrahedraIn',datatype='t',value='@PSDE1.TetrahedraOutliers')
        Tetras = self.tetrahedra
        MyBoxROI.calcTetrasInROI(Tetras)

