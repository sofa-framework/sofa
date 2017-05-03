import Sofa
import ctypes

_dll_path = Sofa.loadPlugin("SofaTest")
_dll = ctypes.CDLL(_dll_path)

_dll.assert_true.restype = None
_dll.assert_true.argtypes = [ctypes.c_bool, ctypes.c_char_p]

_dll.expect_true.restype = None
_dll.expect_true.argtypes = [ctypes.c_bool, ctypes.c_char_p]

_dll.finish.restype = None
_dll.finish.argtypes = []


assert_true = _dll.assert_true
expect_true = _dll.expect_true
finish = _dll.finish

import sys
def handler(type, value, tb):
    try:
        if sys.__excepthook__:
            sys.__excepthook__(type, value, tb)
    except AttributeError as e:
        pass
    
    assert_true(False, "aborting test due to python error: {0}".format(value))

sys.excepthook = handler

