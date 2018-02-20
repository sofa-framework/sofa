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
            
            if (c == "1"):
               #displacement = inputvalue.value[0][0] + 1.
                inputvalue.value = "600 0 0"
               
            elif (c == "2"):
               #displacement = inputvalue.value[0][0] - 1.
               #if(displacement < 0):
               # displacement = 0
                inputvalue.value = "0 0 0"
               
            elif (c == "3"):
                inputvalue2.value = "-600 0 0"
                
            elif (c == "4"):
                inputvalue2.value = "0 0 0"
                
            elif (c == "5"):
                inputvalue3.value = "0 -600 0"
                
            elif (c == "6"):
                inputvalue3.value = "0 0 0"
               
  
