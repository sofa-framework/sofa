import os


import Sofa
Sofa.loadPlugin("Compliant")
from _Compliant import *




def path():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath( os.path.join( current_dir, '..', '..' ) )
