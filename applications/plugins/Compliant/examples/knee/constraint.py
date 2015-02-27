
import mapping
import numpy as np

class PointInHalfSpace(object):

    def __init__(self, node, plane, point, **kwargs):

        self.mapping = mapping.PointPlaneDistance(node, plane, point, **kwargs)

        self.mapping.node.createObject('UniformCompliance', compliance = 0)
        self.mapping.node.createObject('UnilateralConstraint')

        self.stab = self.mapping.node.createObject('Stabilization',
                                                   name = 'stab')
        
        def cb():
            mask = np.array(self.mapping.output.position).flatten() < 0
            mask = np.ones( mask.size ) * mask
            self.stab.mask = str(mask)
        
        self.mapping.script.cb.append( cb )
        
        
