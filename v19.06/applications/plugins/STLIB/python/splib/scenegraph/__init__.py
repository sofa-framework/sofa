# -*- coding: utf-8 -*-
"""
Algorithms we often use.

**Content:**

.. autosummary::

    find
    get

splib.scenegraph.find
*********************
.. autofunction:: find

splib.scenegraph.get
********************
.. autofunction:: get

"""
def find(node, path):
    """
    Query a node or an object by its path from the provided node.

    Example:
        find(node, "/root/rigidcube1/visual/OglModel")
    """
    s = path.split('/')

    if s[1] != node.name:
        return None

    for child in s[2:]:
        newnode = node.getChild(child, warning=False)
        if newnode == None:
            newnode = node.getObject(child)

        if newnode == None:
            return None

        node = newnode
    return node

def get(node, path):
    """
    Query a node, an object or a data by its path from the provided node.

    Example:
        find(node, "/root/rigidcube1/visual/OglModel.position")
    """
    if path.startswith("/"):
        raise Exception("InvalidPathPrefix for " + path+" in "+node.getLinkPath())

    if path.startswith("../"):
        raise Exception("InvalidPathPrefix for " + path+" in "+node.getLinkPath())

    if path.startswith("./"):
        path = path[2:]

    s = path.split('/')
    for child in s:
        newnode = node.getChild(child, warning=False)
        if newnode == None:
            nn = child.split('.')
            if len(nn) == 1:
                newnode = node.getObject(nn[0], warning=False)
                if newnode == None:
                    newnode = node.getData(nn[0])
            elif len(nn) == 2:
                newnode = node.getObject(nn[0]).getData(nn[1])

        if newnode == None:
            raise Exception("Invalid Path Query for "+path+" in "+node.getLinkPath())

        node = newnode
    return node

def getLinkPath(node, path):
    return get(node,path).getLinkPath()

