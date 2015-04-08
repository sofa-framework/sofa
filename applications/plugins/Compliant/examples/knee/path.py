import Sofa

def split(path):
    return path.lstrip('@').split('/')

def get_node(start, path):
    '''node from path'''
    
    if type(path) is str:
        return get_node(start, split(path))

    res = start
    
    for d in path:
        if d == '..':
            res = res.getParent()
        else:
            res = res.getChild(d)
            
    return res

def get_object(start, path):
    '''object from path'''

    if type(path) is str:
        return get_object(start, split(path))
    else:
        return get_node(start, path[:-1]).getObject(path[-1])
    
    
def relative( start, end ):
    '''relative path between objects/nodes'''
    
    path = {}
    depth = {} 
    
    for i in [start, end]:
        if type(i) == Sofa.Node:
            path[i] = i.getPathName()
        else:
            path[i] = i.getContext().getPathName() + '/' + i.name

        # discard initial empty chunk
        path[i] = split(path[i])[1:]


        depth[i] = len( path[i] ) if type(i) is Sofa.Node else len(path[i]) - 1

    # TODO don't go up until root, find common prefix instead ?
    return '/'.join( ( ['..'] * depth[start] ) + path[end] )
