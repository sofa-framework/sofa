
import Sofa

from Compliant import StructuralAPI as api

import ctypes


class Event(ctypes.Structure):
    # the vtable stuff is of course not portable. the best fix would
    # be to expose offsetof(Event::m_handled) somehow and pack bytes
    # accordingly.
    _fields_ = (('__vtable__', ctypes.c_void_p),
                ('handled', ctypes.c_bool))
    
    
class SolveEndEvent(Event):
    _fields_ = (('index', ctypes.c_size_t) ,)

    

class Script(api.Script):

    def onEvent(self, name, ptr):
        if name == 'SolveEndEvent':
            event = SolveEndEvent.from_address(ptr)
            print(event.index)
            event.handled = True
            
            
def createScene(node):
    node.createObject('CompliantImplicitSolver', extra_solves = 5)
    node.createObject('LDLTSolver')
    
    script = Script(node)
    script.receive_all_events = True
    
