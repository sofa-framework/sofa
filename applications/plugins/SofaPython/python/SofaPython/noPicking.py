import sys
import Sofa
import Tools


class Visitor(object):
## add tag 'NoPicking' to every mstates
## so picking is only performed by collision detection on CollisionModel

    def treeTraversal(self):
        #print 'ValidationVisitor treeTraversal'
        return -1 # dag

    def processNodeTopDown(self,node):

        state = node.getMechanicalState()
        if state is not None:
            tags = state.tags
            tags.append( ['NoPicking'] )
            sys.stdout.flush()
            state.tags = Tools.listListToStr(tags)

        return True

    def processNodeBottomUp(self,node):
        return True


def removeMStatePicking(node):
    ## every mstates under node are no longer pickable
    ## only CollisionModel picking will be available

    vis = Visitor()
    node.executeVisitor(vis)
