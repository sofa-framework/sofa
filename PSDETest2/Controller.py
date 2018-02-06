#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Sofa

class Controller(Sofa.PythonScriptController):

    def initGraph(self, node):
            self.node = node

    def onKeyPressed(self,c):
  
            #self.node.getRoot().getChild('model').removeObject('boxROITest') # WARNING: removing any object (apparently) at runtime will cause segfault
  
            inputvalue = self.node.getChild('constantForce').getObject('forceField').findData('forces')
            inputvalue2 = self.node.getChild('constantForce2').getObject('forceField2').findData('forces')
            inputvalue3 = self.node.getChild('constantForce3').getObject('forceField3').findData('forces')
            #inputvalue2 = []            
       
            elif (c == "1"):
                inputvalue2.value = "-30000 0 0"
                
            elif (c == "2"):
                inputvalue2.value = "0 0 0"                               
  
