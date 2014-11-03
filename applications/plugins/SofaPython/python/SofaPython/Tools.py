import Sofa

def listToStr(x):
    """ concatenate lists for use with data.
    """
    return ' '.join(map(str, x))

def getNode(rootNode, path):
    """ Return node at path or None if not found
    """
    currentNode = rootNode
    pathComponents = path.split('/')
    for c in pathComponents:
        if len(c)==0: # for leading '/' and in case of '//'
            continue
        currentNode = currentNode.getChild(c)
        if currentNode is None:
            print "SofaPython.Tools.findNode: can't find node at", path
            return None
    return currentNode
