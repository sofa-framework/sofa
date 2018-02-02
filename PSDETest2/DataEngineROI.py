#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI



class MyDataEngine(Sofa.PythonScriptDataEngine):       
    
    def parse(self):        
        # this is the ideal place to define inputs and outputs!
        
        self.addNewInput('Positions',datatype='p',value='@tetras.position')        
        self.addNewInput('Tetrahedra',datatype='t',value='@container.tetrahedra')        
        self.addNewOutput('TetrahedraInliers',datatype='t')                
        self.addNewOutput('TetrahedraOutliers',datatype='t')        
        

    def update(self):
        
        print self.Positions[0]        
        MyBoxROI = BoxROI(-40, -30, 60, -25, 30, 100, self.Positions)            
        MyBoxROI.calcTetrasInROI(self.Tetrahedra)               
        self.TetrahedraInliers = MyBoxROI.Inliers        
        self.TetrahedraOutliers = MyBoxROI.Outliers

    def init(self):        
        pass
        #self.addNewOutput('TetrahedraOutliers',blup='sd',blep='sd',datatype='t') # works here ... really, really strange
        
   
   
