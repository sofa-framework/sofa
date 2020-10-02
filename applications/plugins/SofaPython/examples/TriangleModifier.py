import Sofa

class TriangleModifier(Sofa.PythonScriptController):
    def initGraph(self,node):
        self.node = node
        self.triangleModifier = self.node.getObject("triangleModifier")
        self.nbSteps = 0
        return 0
    
    def onEndAnimationStep(self,dt):
        if self.nbSteps == 0:
            self.triangleModifier.addTriangles( [ (0,1,2), (1,3,2), (1,4,3) ] )
        if self.nbSteps == 1:
            pointAncestor = Sofa.PointAncestorElem()
            pointAncestor.type = 2
            pointAncestor.localCoords = (0.33, 0.33, 0 )
            pointAncestor.index = 0
            
            self.triangleModifier.addPoints( [ pointAncestor ] )
        
        if self.nbSteps == 2:
            self.triangleModifier.addRemoveTriangles( [ (0,1,5), (1,2,5), (2,0,5) ], [ 4,5,6 ], [ [1], [1], [1] ], [ [1], [1], [1] ], [ 0 ]   ) 
            
        self.nbSteps+=1
        return 0
   
   