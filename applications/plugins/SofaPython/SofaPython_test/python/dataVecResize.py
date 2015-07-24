import Sofa
import SofaPython.Tools
import SofaTest


def createScene(node):
    ## testing Data<vector<T>> resize from python binding

    # dummy components, just to use their Data<vector<T>>
    dof = node.createObject("MechanicalObject",template="Vec3d",name="dof",position="0 0 0  1 1 1  2 2 2  3 3 3  4 4 4  5 5 5")
    fc = node.createObject("FixedConstraint", name="fc", indices="0 1 2 3 4")
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')



class VerifController(SofaTest.Controller):

    def onEndAnimationStep(self,dt):

        self.dof = self.node.getObject("dof")
        self.fc = self.node.getObject("fc")

        # two-dimensional vector resize

        self.ASSERT( len(self.dof.position)==6, "test1" )
        self.dof.position=[9,9,9] # smaller
        self.ASSERT( len(self.dof.position)==1, "test2" )
        self.dof.position=[] # empty list
        self.ASSERT( len(self.dof.position)==0, "test3" )
        self.dof.position=[9,9,9]*9 # larger
        self.ASSERT( len(self.dof.position)==9, "test4" )
        self.dof.position="" # empty string
        self.ASSERT( len(self.dof.position)==0, "test5" )
        self.dof.position=SofaPython.Tools.listToStr( [9,9,9]*9 ) # larger string
        self.ASSERT( len(self.dof.position)==9, "test6" )



        # one-dimensional vector resize

        self.ASSERT( len(self.fc.indices)==5, "test7" )
        self.fc.indices=[8]
        self.ASSERT( len(self.fc.indices)==1, "test8" )
        self.fc.indices=[]
        self.ASSERT( len(self.fc.indices)==0, "test9" )
        self.fc.indices=[0,1,6,3,4,5,3,1,2]
        self.ASSERT( len(self.fc.indices)==9, "test10" )
        self.fc.indices=""
        self.ASSERT( len(self.fc.indices)==0, "test11" )
        self.fc.indices="0 1 6 3 4 5 3 1 2"
        self.ASSERT( len(self.fc.indices)==9, "test12" )
        self.fc.indices="8"
        self.ASSERT( len(self.fc.indices)==1, "test13" )

        self.sendSuccess()