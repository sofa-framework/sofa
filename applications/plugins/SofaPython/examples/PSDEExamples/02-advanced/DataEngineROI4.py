#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine4(Sofa.PythonScriptDataEngine):       
    
    def parse(self):        
        # this is the ideal place to define inputs and outputs!
        
        self.addNewInput('Positions',datatype='p',value='@tetras.rest_position')        
        self.addNewInput('Tetrahedra',datatype='t',value='@PSDE3.TetrahedraOutliers')        
        self.addNewOutput('TetrahedraInliers',datatype='t')                
        self.addNewOutput('TetrahedraOutliers',datatype='t')        

    def update(self):
	pass

    def init(self):                
        
        MyBoxROI = BoxROI(1.3, -1, -1.2, 3, 1, -0.2, self.Positions)
        MyBoxROI.calcTetrasInROI(self.Tetrahedra)        
        self.TetrahedraInliers = MyBoxROI.Inliers        
        self.TetrahedraOutliers = MyBoxROI.Outliers
        



