from __future__ import print_function

import os
import Sofa
Sofa.loadPlugin("Compliant")
from _Compliant import *




def path():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath( os.path.join( current_dir, '..', '..' ) )



def set_compliant_interactor(compliance = 1e-3):

    import ctypes
    plugin_name = 'compliant_qtquickgui'

    try:
        dll_path = Sofa.loadPlugin(plugin_name)

        dll = ctypes.CDLL(dll_path)

        dll.set_compliant_interactor.argtypes = [ctypes.c_double]
        dll.set_compliant_interactor.restype = None

        print('setting compliant interactor, compliance:', compliance)
        dll.set_compliant_interactor(compliance)
        
    except EnvironmentError as e:
        print('setting compliant interactor failed:', e)
        
