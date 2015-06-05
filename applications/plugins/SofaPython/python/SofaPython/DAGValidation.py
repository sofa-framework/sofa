import sys
import Sofa


        


class Visitor(object):
    
        def __init__(self):
            #print "DAGValidationVisitor"
            self.error = []

        def treeTraversal(self):
            #print 'ValidationVisitor treeTraversal'
            return -1 # dag
            
        def processNodeTopDown(self,node):
            mapping = node.getMechanicalMapping()
            #print node.name
            
            if not mapping is None:
                #print mapping.getName()
                from_dof = mapping.getFrom()
                parent_node = mapping.getContext().getParents()
                
                parent_node_path = []
                for p in parent_node:
                    parent_node_path.append( p.getPathName() )
                    
                from_node_path = []
                for f in from_dof:
                    from_node_path.append( f.getContext().getPathName() )
                #print parent_node_path
                
                for f in from_node_path:
                    #print f
                    if not f in parent_node_path:
                        err = "ERROR "
                        err += "'"+mapping.getContext().getPathName()+"/"+mapping.name+"': "
                        err += "'"+ f + "' should be a parent node"
                        self.error.append(err)
                        #print err
                        
                for p in parent_node_path:
                    #print p
                    if not p in from_node_path:
                        err = "ERROR "
                        err += "'"+mapping.getContext().getPathName()+"/"+mapping.name+"': "
                        err += "'"+p + "' should NOT be a parent node"
                        self.error.append(err)
                        #print err     
                        
            #print "==================="
            return True
            
        def processNodeBottomUp(self,node):
            return True
        
        
        
        
def test( node ):
    
    print ""
    print "====== SofaPython.DAGValidation.test ======================="
    print ""
    print "Validating scene from node '/" + node.getPathName() + "'..."
    
    vis = Visitor()
    node.executeVisitor(vis)
    
    if len(vis.error) is 0:
        print "... VALIDATED"
    else:
        print "... NOT VALID"
        print ""
        for e in vis.error:
            print e
            
    print ""
    print "=============================================================="
    sys.stdout.flush()