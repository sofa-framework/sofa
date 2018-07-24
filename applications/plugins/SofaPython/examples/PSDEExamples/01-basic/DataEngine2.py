class MyDataEngine(Sofa.PythonScriptDataEngine):           
  
    def parse(self):        
        # this is the correct place to define inputs and outputs! (linking will not work in init() for instance)
        self.addNewInput('NumPoints', value='@PSDE.NumPoints')       
        self.addNewOutput('NumPoints2',datatype='int')                

    def init(self):
	pass

    def update(self):        
        self.NumPoints2 = self.NumPoints
        
