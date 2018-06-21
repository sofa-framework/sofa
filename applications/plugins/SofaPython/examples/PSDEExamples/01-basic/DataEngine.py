import Sofa
import os

class MyDataEngine(Sofa.PythonScriptDataEngine):
  
    def parse(self):        
        # this is the correct place to define inputs and outputs! (linking will not work in init() for instance)
        self.addNewInput('positions', value='@loader.position')
        self.addNewOutput('firstPos', value="0")
        
    def init(self):
	pass
    
    def update(self):
        value = [0, 0, 0]
        for pos in self.positions:
            value[0] += pos[0]
            value[1] += pos[1]
            value[2] += pos[2]
        value[0] /= len(self.positions)
        value[1] /= len(self.positions)
        value[2] /= len(self.positions)
        self.firstPos = value
        pass
