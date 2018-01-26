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
            
            ##awa3 = self.node.getChild('model').getChild('modelSubTopoS6').getObject('containerS6').findData('tetrahedra')            
            
            ## find total number of tetras
            #TetrasAll = self.node.getChild('model').getObject('loader').findData('tetrahedra')
            #TotalNTetras = len(TetrasAll.value)           
            #print TetrasAll.value[0]
            
            ## find the union of all tetras if ROIs representing sensors
            #TotalROIIndices=[]
            #NROIs = 6
            #for i in range(1,NROIs+1):
                #ObjectName = 'boxROISubTopoS' + str(i)
                #TetraIndicesInROI = self.node.getChild('model').getObject(ObjectName).findData('tetrahedronIndices')
                ##print TetraIndicesInROI.value[1:len(TetraIndicesInROI.value)][0]                
                #TotalROIIndices= TotalROIIndices+TetraIndicesInROI.value                
        
            #TotalROIIndicesStripped = []
            #for i in range(0,len(TotalROIIndices)-1):
                #TotalROIIndicesStripped.append(TotalROIIndices[i][0])
            
            #TotalROIIndicesStripped.sort()            
                
            ## build the complementary index set to the set above
            #ComplementROIIndices = []
            #CurrentStartIdx = 0
            #for i in TotalROIIndicesStripped:
                #ComplementROIIndices = ComplementROIIndices + range(CurrentStartIdx, i)
                #CurrentStartIdx = i+1      
            
            #ComplementROIIndices = ComplementROIIndices + range(CurrentStartIdx,TotalNTetras)
                            
            ##print TotalROIIndicesStripped
            
            ##print '....'
            
            ##print ComplementROIIndices
            
            
            
            #TetrasSofter = [TetrasAll.value[i] for i in TotalROIIndicesStripped]
            #TetrasHard = [TetrasAll.value[i] for i in ComplementROIIndices]
            
            ##print TetrasSofter
            
            
            ## Init the rest of the scene now that we have the correct tetrahedronIndices
            #model = self.node.getChild('model')                        
     
     
            
            
            ##self.node.getChild('model').createObject('BoxROI', name='boxROITest', box='100 -100 100 -20 10 30', drawBoxes='true', position="@tetras.rest_position", tetrahedra="@container.tetrahedra") # WARNING: possible to create object after init, but gives out a warning
            
            
            
            ##print('asda')
            ##print(awa3.value)
            ##print(awa2.value[0])
