class MyDataEngine(Sofa.PythonScriptDataEngine):           
  
    def parse(self):        
        # this is the correct place to define inputs and outputs! (linking will not work in init() for instance)
        self.addNewInput('Positions',datatype='vector<Vec3d>',value='@loader.position') # even though it is not strictly necessary, because in this case it can be deduced, the type of Data can be declared explicitly
        self.addNewOutput('NumPoints',datatype='int') # explicitly declaring the output type allows the chaining of PSDEs.               

    def init(self):
	pass

    def update(self):        
        self.NumPoints = len(self.Positions)
