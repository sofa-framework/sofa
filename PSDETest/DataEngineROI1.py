#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine(Sofa.PythonScriptDataEngine):

    def update(self):
        print 'blup'        

    def init(self):
        MyBoxROI = BoxROI(-30, -30, 8, 0, 30, 50, self.position)
        Tetras = self.tetrahedra
        MyBoxROI.calcTetrasInROI(Tetras)
	self.addNewOutput('TetrahedraInliers',datatype='t',value=MyBoxROI.Inliers)
	self.addNewOutput('TetrahedraOutliers',datatype='t',value=MyBoxROI.Outliers)



