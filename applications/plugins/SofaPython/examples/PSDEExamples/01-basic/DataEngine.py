class MyDataEngine(Sofa.PythonScriptDataEngine):           
  
    def parse(self):        
        # this is the correct place to define inputs and outputs! (linking will not work in init() for instance)        
        self.addNewInput('Positions',datatype='p',value='@loader.position')       
        self.addNewOutput('NumPoints',datatype='d', value='0')                

    def init(self):
	pass

    def update(self):        
        self.NumPoints = len(self.Positions)
