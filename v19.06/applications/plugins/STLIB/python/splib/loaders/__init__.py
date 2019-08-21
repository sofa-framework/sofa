# -*- coding: utf-8 -*-
"""
Utilitary function to ease the writing of scenes.

.. autosummary::

    loadPointListFromFile
    getLoadingLocation

splib.loaders.loadPointListFromFile
***********************************
.. autofunction:: loadPointListFromFile
.. autofunction:: getLoadingLocation

"""
__all__=[]

import os
def getLoadingLocation(filename , source=None):
    """Compute a loading path for the provided filename relative to a given 
       source location
       
       Examples:
           getLoadingLocation("myfile.json")   # returns "myfile.json
           getLoadingLocation("myfile.json", "toto") #returns "/fullpath/to/toto/myfile.json"
           getLoadingLocation("myfile.json", __file__) #returns "/fullpath/to/toto/myfile.json"

        The latter is really usefull to make get the path for a file relative
        to the 'current' python source.                    
    """
    if source == None:
        return filename
    
    if os.path.isfile(source):
        source = os.path.dirname(os.path.abspath(source))
    
    return os.path.join(source, filename)


def loadPointListFromFile(s):
    """Load a set of 3D point from a json file"""
    import json
    return json.load(open(s))
