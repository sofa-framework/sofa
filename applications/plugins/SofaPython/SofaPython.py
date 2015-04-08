import os
import os.path
import sys
import locale

import Sofa



# add plugin path to sys.path
plugins_paths = ['applications/plugins',
                 'applications-dev/plugins']

for relative in plugins_paths:
    absolute = os.path.join(Sofa.src_dir(), relative)
    if 'SOFA_PythonScriptPath' in locals() and len(SOFA_PythonScriptPath) > 0 :
        absolute=os.path.join(os.path.abspath(SOFA_PythonScriptPath),relative)
    #print absolute
    if os.path.exists(absolute): # applications-dev is not necessarily existing
        for plugin in os.listdir( absolute ):
            path = os.path.join(absolute, plugin)
            if os.path.isdir( path ):
                python = os.path.join(path, 'python')
                if os.path.exists( python ):
                    print "SofaPython: added plugin path for", plugin
                    sys.path.insert(0, python )
                    #print "SofaPython: located in ", python

# add more customization here if needed

# force C locale
locale.setlocale(locale.LC_ALL, 'C')

# try to import numpy and to launch numpy.finfo to cache data (avoid a deadlock when calling numpy.finfo from a worker thread)
try:
    import numpy
    numpy.finfo(float)
except:
    pass
