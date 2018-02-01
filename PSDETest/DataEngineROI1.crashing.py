#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI

class MyDataEngine(Sofa.PythonScriptDataEngine):    

    def parse(self):        
        # this is the ideal place to define inputs and outputs!
        
        self.addNewInput('Positions',datatype='p',value='@tetras.rest_position')        
        self.addNewInput('Tetrahedra',datatype='t',value='@container.tetrahedra')        
        self.addNewOutput('TetrahedraInliers',datatype='t')                
        self.addNewOutput('TetrahedraOutliers',datatype='t')
        
        
        #
	

    def update(self):
	pass

    def init(self):        
        #self.addNewOutput('TetrahedraOutliers',blup='sd',blep='sd',datatype='t') # works here ... really, really strange
        MyBoxROI = BoxROI(-30, -30, 8, 0, 30, 50, self.Positions)    
        MyBoxROI.calcTetrasInROI(self.Tetrahedra)
        self.TetrahedraInliers = MyBoxROI.Inliers
        self.TetrahedraOutliers = MyBoxROI.Outliers
	



