import sys
import Sofa
import Tools


def MechanicalObjectVisitor(node):
## listing mechanical states, bottom-up from node
        ancestors = []
        visited = []

        for p in node.getParents():
            path = p.getPathName()
            if not path in visited:
                state = p.getMechanicalState()
                if not state is None:
                    ancestors.append( path+"/"+state.name )
                ancestors += MechanicalObjectVisitor( p )

        return ancestors



class Visitor(object):
## checking that mapping graph is equivalent to node graph
## checking that independent dofs are not under other dofs in the scene graph
    def __init__(self):
        #print "DAGValidationVisitor"
        self.error = []

    def treeTraversal(self):
        #print 'ValidationVisitor treeTraversal'
        return -1 # dag

    def processNodeTopDown(self,node):

        #print node.name

        state = node.getMechanicalState()

        if state is None:
            return True

        mapping = node.getMechanicalMapping()


        if mapping is None: #independent dofs

            ancestors = MechanicalObjectVisitor(node)
            if not len(ancestors) is 0: # an independent dof is under other dofs in the scene graph
                err = "ERROR "
                err += "mechanical state '"+state.getContext().getPathName()+"/"+state.name+"' is independent (no mapping)"
                err += " and should not be in the child node of other mechanical states ("+Tools.listToStr(ancestors)+")"
                self.error.append(err)

        else: # mapped dofs

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
        
        
        
        
def test( node, silent=False ):
## checking that mapping graph is equivalent to node graph
## checking that independent dofs are not under other dofs in the scene graph
## return a list of errors

    if not silent:
        print ""
        print "====== SofaPython.DAGValidation.test ======================="
        print ""
        print "Validating scene from node '/" + node.getPathName() + "'..."
    
    vis = Visitor()
    node.executeVisitor(vis)
    
    if not silent:
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

    return vis.error
