#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
        
class MyDataEngineUnion2(Sofa.PythonScriptDataEngine):        
    
    def parse(self):        
        # this is the ideal place to define inputs and outputs!
        
        self.addNewInput('Tetras1',datatype='t',value='@PSDE3.TetrahedraInliers')        
        self.addNewInput('Tetras2',datatype='t',value='@PSDE4.TetrahedraInliers')        
        self.addNewOutput('TetrahedraUnion',datatype='t')                        

    def update(self):
	pass

    def init(self):                
        # warning! Assumes tetra sets are disjunct!
        self.TetrahedraUnion = self.Tetras1 + self.Tetras2
        






