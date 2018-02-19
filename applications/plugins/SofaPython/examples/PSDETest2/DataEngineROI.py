#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from ROI import BoxROI



class MyDataEngine(Sofa.PythonScriptDataEngine):       
    
    def parse(self):        
        # this is the correct place to define inputs and outputs! (linking will not work in init() for instance)        
        
        self.addNewInput('Positions',datatype='p',value='@tetras.position')        
        self.addNewInput('Tetrahedra',datatype='t',value='@container.tetrahedra')        
        self.addNewOutput('TetrahedraInliers',datatype='t')                
        self.addNewOutput('TetrahedraOutliers',datatype='t')       

    def update(self):        

        MyBoxROI = BoxROI(-40, -30, 60, -25, 30, 100, self.Positions)            
        MyBoxROI.calcTetrasInROI(self.Tetrahedra)               
        self.TetrahedraInliers = MyBoxROI.Inliers        
        self.TetrahedraOutliers = MyBoxROI.Outliers
        print 'Number of Tetras in ROI:' 
	print len(self.TetrahedraInliers)

    def init(self):        
        pass

        
   
   
