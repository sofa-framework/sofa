import Sofa

import inspect
import os


# a controller to return a test result directly from a python script
class Controller(Sofa.PythonScriptController):

    def onLoaded(self, node):
        self.node = node
        self.root = node.getRoot()
        
    def should(self, value, msg = 'unknown'):
        if value:
            self.node.sendScriptEvent('success', 0)
        else:
            callerframerecord = inspect.stack()[1]

            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)

            f = open(info.filename)
            lines = f.readlines()

            # print
            print '{0}:{1}: {2}'.format(os.path.abspath(info.filename),
                                            info.lineno, 'Failure')
            print 'Reason:', msg
            # print #lines[ info.lineno - 1 ]

            self.node.sendScriptEvent('failure', 0)
        self.root.findData('animate').value = 0
            

