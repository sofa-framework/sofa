#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa

 
    
        

class MyDataEngineIntersection(Sofa.PythonScriptDataEngine):    
    
    def calcIntersection(self,Tetras1,Tetras2):
        Intersection = []
        while len(Tetras1)!=0:                       
            for i in range(0,len(Tetras2)):                
                if Tetras1[0] == Tetras2[i]:                   
                    Intersection = Intersection + Tetras2[i]      
                    break
            del(Tetras1[0])        
                    
        return Intersection   

    def parse(self):        
        # this is the ideal place to define inputs and outputs!
        
        self.addNewInput('Tetras1',datatype='t',value='@PSDE1.TetrahedraOutliers1')        
        self.addNewInput('Tetras2',datatype='t',value='@PSDE1.TetrahedraOutliers2')        
        self.addNewOutput('TetrahedraIntersection',datatype='t')                        

    def update(self):
	pass

    def init(self):        
        #TotalTetras = self.Tetras1+self.Tetras2
        self.TetrahedraIntersection = self.calcIntersection(self.Tetras1, self.Tetras2)
        






