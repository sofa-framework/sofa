# -*- coding: utf-8 -*-
import Sofa

class BoxFilter(object):
        def __init__(self, node, sourceObject, orientedBoxes):      
                node = node.createChild("Select")          
                self.sourceObject = sourceObject
                self.boxes = []
                for orientedBox in orientedBoxes:
                        box = node.createObject("BoxROI", name="filters",
                               orientedBox=orientedBox,
                               position=sourceObject.dofs.findData("position").getLinkPath(), 
                               drawBoxes=True, drawPoints=True, drawSize=2.0)
                        self.boxes.append(box)
                       
        def getIndices(self):
                self.sourceObject.init()
                indices = []
                for box in self.boxes:
                        box.init()
                        indices.append(map(lambda x: x[0], box.indices))
                return indices

