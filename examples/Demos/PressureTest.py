import Sofa
import math

resolutionCircumferential=7
resolutionRadial=3
resolutionHeight=7
maxIteration=3000

vIndex=int((resolutionCircumferential*(resolutionRadial-1)+1)*resolutionHeight/2)


youngModulus = 1.0
poissonRatio = 0.499999
pressure = 0.6


class LongitudinalDefComputation(Sofa.Core.Controller):

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.node = kwargs["node"]


    def init(self):
        self.p0x = self.node.state.position.value[vIndex][0]
        self.p0y = self.node.state.position.value[vIndex][1]
        self.p0z = self.node.state.position.value[vIndex][2]
        self.step = 0
        pass

    def onAnimateEndEvent(self, event):
        self.step += 1
        self.p1x = self.node.state.position.value[vIndex][0]
        self.p1y = self.node.state.position.value[vIndex][1]
        self.p1z = self.node.state.position.value[vIndex][2]
        longitudinalDeformation=(self.p1z-self.p0z)/self.p0z
        goalLongitudinalDef = pressure/youngModulus

        print(f"Step {self.step}")
        print(f"Current longitudinal deformation at id {vIndex} : {longitudinalDeformation}, expected one : {goalLongitudinalDef}")
        print(f"Difference = {longitudinalDeformation - goalLongitudinalDef}")
        print(f"")

        radialDef=(self.p0x*self.p1x + self.p0y*self.p1y)/(self.p0x*self.p0x + self.p0y*self.p0y) - 1
        goalRadialDef=-pressure*poissonRatio/youngModulus

        print(f"Current radial deformation at id {vIndex} : {radialDef}, expected one : {goalRadialDef}")
        print(f"Difference = {radialDef - goalRadialDef}")
        print(f"------------------------------")

def createScene(node):
    node.dt = 0.05
    node.gravity = [0, 0, 0]
    node.addObject("VisualStyle", displayFlags="showBehaviorModels showForceFields showInteractionForceFields showCollisionModels showMappings")

    node.addObject("DefaultAnimationLoop", name = "animationLoop" )
    cylinder = node.addObject("GenerateCylinder", name = "cylinder", radius = 0.2, height = 1.0,
                   resCircumferential=resolutionCircumferential,
                   resRadial=resolutionRadial,
                   resHeight=resolutionHeight)

    node.addObject("TetrahedronSetTopologyContainer", name="Container1", createTriangleArray = True,
                   tetrahedra = cylinder.tetrahedra.linkpath,
                   position = cylinder.output_TetrahedraPosition.linkpath)
    node.addObject("TetrahedronSetGeometryAlgorithms")


    node.addObject("CGLinearSolver", name = "linearSolver",
                   iterations = maxIteration,
                   tolerance = 1e-9,
                   threshold = 1e-9)

    node.addObject("StaticEquilibriumIntegrationScheme", name = "StaticSolver",
                   printLog = True,
                   maxNbIterationsNewton = 20,
                   maxNbIterationsLineSearch = 6,
                   lineSearchReductionRate=0.2,
                   newtonStepSize = 2.0,
                   alwaysAdvanceNewton = True)


    mstate = node.addObject("MechanicalObject", name="state",  position = cylinder.output_TetrahedraPosition.linkpath)

    node.addObject("TetrahedralCorotationalFEMForceField",
                    youngModulus = youngModulus,
                    poissonRatio = poissonRatio,
                    method = "large")

    node.addObject("MeshMatrixMass", name = "BezierMass",
                   massDensity = 1.0,
                   lumping=False)

    boxROI = node.addObject("BoxROI", name = "boxRoiFix",
                   box = [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01],
                   strict = False)

    node.addObject("FixedProjectiveConstraint",
                   indices = boxROI.indices.linkpath)

    node.addObject("FixedPlaneProjectiveConstraint",
                   dmin= -0.01,
                   dmax= 0.01,
                   direction= [0,0,1])


    boxROIPressure = node.addObject("BoxROI", name = "boxRoiPressure",
                            box = [-0.2, -0.2, 0.99, 0.2, 0.2, 1.01],
                            computeTriangles=True,
                            strict = False)

    node.addObject("TrianglePressureForceField",
                   triangleList = boxROIPressure.triangleIndices.linkpath,
                   pressure = [0, 0, pressure])

    node.addObject("LineProjectiveConstraint",
                   direction=[1, 0, 0],
                   origin=[0, 0, 0],
                   indices = [resolutionCircumferential*(resolutionRadial-1)+1])

    node.addObject(LongitudinalDefComputation(node = node))


