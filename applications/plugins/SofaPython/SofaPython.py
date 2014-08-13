import os
import os.path
import sys
import Sofa


# add plugin path to sys.path
plugins_paths = ['applications/plugins',
                 'applications-dev/plugins']

for relative in plugins_paths:
    absolute = os.path.join(Sofa.src_dir(), relative)
    if os.path.exists(absolute): # applications-dev is not necessarily existing
        
        for plugin in os.listdir( absolute ):
            path = os.path.join(absolute, plugin)
            if os.path.isdir( path ):
                python = os.path.join(path, 'python')
                if os.path.exists( python ):
                    print "SofaPython: added plugin path for", plugin
                    sys.path.append( python )

# add more customization here if needed
