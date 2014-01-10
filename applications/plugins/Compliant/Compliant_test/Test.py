import Sofa

# some helpers
class Controller(Sofa.PythonScriptController):

    def onLoaded(self, node):
        self.node = node
        self.root = node.getRoot()
        
    def should(self, value):
        if value:
            self.node.sendScriptEvent('success', 0)
        else:
            self.node.sendScriptEvent('failure', 0)
