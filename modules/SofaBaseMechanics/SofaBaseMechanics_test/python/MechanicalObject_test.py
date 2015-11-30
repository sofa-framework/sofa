import Sofa
import SofaTest



def createScene(node):

    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')



class VerifController(SofaTest.Controller):

    def initGraph(self,node):

        testnode = self.node.createChild("testnode")

        # default
        self.dof = testnode.createObject('MechanicalObject', template="Vec3d", name="dof1")

        # given pos and vel
        self.dof2 = testnode.createObject('MechanicalObject', template="Vec3d", name="dof2", position="0 0 0  1 1 1 ", velocity="0 0 0  0 0 0")

        # given pos
        self.dof3 = testnode.createObject('MechanicalObject', template="Vec3d", name="dof3", position="0 0 0  1 1 1 ")

        # given vel
        self.dof4 = testnode.createObject('MechanicalObject', template="Vec3d", name="dof4", velocity="0 0 0  1 1 1 ")

        # given rest-pos
        self.dof41 = testnode.createObject('MechanicalObject', template="Vec3d", name="dof41", rest_position="0 0 0  1 1 1")


        # explicitly linked to empty vector
        testnode.createObject('MeshTopology', position="", name="topology" ) # empty topology
        self.dof5 = testnode.createObject('MechanicalObject', template="Vec3d", name="dof5", position="@topology.position")

        # implicitly defined from topology
        topologynode = self.node.createChild("fromtopoloy")
        topologynode.createObject('MeshTopology', position="0 0 0 1 1 1 2 2 2", triangles="0 1 2" ) # empty topology
        self.dof6 = topologynode.createObject('MechanicalObject', template="Vec3d", name="dof6")






    def onBeginAnimationStep(self,dt):

        # testing only when everything is initialized

        self.ASSERT(len(self.dof.position)==1, "test1 default position size" )
        self.ASSERT(len(self.dof.velocity)==1, "test1 default velocity size" )

        self.ASSERT(len(self.dof2.position)==2, "test2 given position size" )
        self.ASSERT(len(self.dof2.velocity)==2, "test2 given velocity size" )

        self.ASSERT(len(self.dof3.position)==2, "test3 given position size" )
        self.ASSERT(len(self.dof3.velocity)==2, "test3 default velocity size" )

        self.ASSERT(len(self.dof4.position)==2, "test4 default position size" )
        self.ASSERT(len(self.dof4.velocity)==2, "test4 given velocity size "+str(len(self.dof4.velocity)) )

        self.ASSERT(len(self.dof41.position)==2, "test41 from rest pos "+str(len(self.dof41.position)) )
        self.ASSERT(len(self.dof41.velocity)==2, "test41 default velocity size "+str(len(self.dof41.velocity)) )

        self.ASSERT(len(self.dof5.position)==0, "test5 given empty position size "+str(len(self.dof5.position)) )
        self.ASSERT(len(self.dof5.velocity)==0, "test5 default velocity size "+str(len(self.dof5.velocity)) )

        self.ASSERT(len(self.dof6.position)==3, "test6 implicit topology position size "+str(len(self.dof6.position)) )
        self.ASSERT(len(self.dof6.velocity)==3, "test6 default velocity size "+str(len(self.dof6.velocity)) )


        self.sendSuccess()
