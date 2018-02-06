#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa

class Controller(Sofa.PythonScriptController):

    def initGraph(self, node):
            self.node = node

    def onKeyPressed(self,c):
  
            #self.node.getRoot().getChild('model').removeObject('boxROITest') # WARNING: removing any object (apparently) at runtime will cause segfault
  
            inputvalue = self.node.getChild('constantForce').getObject('forceField').findData('forces')

 
            if (c == "1"):
                inputvalue.value = "-30000 0 0"
                
            elif (c == "2"):
                inputvalue.value = "0 0 0"                
               
  
