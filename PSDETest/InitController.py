#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa
from datetime import datetime

class InitController(Sofa.PythonScriptController):

    def initGraph(self, node):
            self.node = node

    def bwdInitGraph(self,node):            
            
            print(str(datetime.now()))
            
            A = Sofa.LinearSpring(0,1,4,5,60)
            
            modelSubTopoS1 = self.node.getChild('model').getChild('modelSubTopoS1')
            YM = modelSubTopoS1.getObject('FEMS1').findData('youngModulus')
            YM = 200
            
           
