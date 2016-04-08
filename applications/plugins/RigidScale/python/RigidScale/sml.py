import Sofa
import RigidScale.API
import SofaPython.sml
import Compliant.StructuralAPI
import Compliant.sml

printLog = True

def insertRigidScale(parentNode, solidModel, param):
    """ create a RigidScale.API.ShearlessAffineBody from the solidModel
    """
    if printLog:
        Sofa.msg_info("RigidScale.sml", "insertRigidScale "+solidModel.name)

    body = RigidScale.API.ShearlessAffineBody(parentNode, solidModel.name)

#    massinfo = SofaPython.sml.getSolidRigidMassInfo(rigidModel, density)
#    body.setFromRigidInfo(massinfo, offset=solidModel.position , inertia_forces = False )

    if (not len(solidModel.mesh)==1):
        Sofa.msg_warning("RigidScale.sml", "insertRigidScale support only single mesh solid (nb meshes={0}) - solid {1} ignored".format(len(solidModel.mesh), solidModel.name))
        return None

    body.setFromMesh(solidModel.mesh[0].source, voxelSize=SofaPython.units.length_from_SI(param.voxelSize), density=SofaPython.units.massDensity_from_SI(1000.), offset=solidModel.position)
    body.addElasticBehavior("behavior", stiffness=SofaPython.units.elasticity_from_SI(param.rigidScaleStiffness), poissonCoef=0, numberOfGaussPoint=8)
    cm = body.addCollisionMesh(solidModel.mesh[0].source, offset=solidModel.position)
    cm.addVisualModel()

    body.affineDofs.showObject=param.showAffine
    body.affineDofs.showObjectScale=SofaPython.units.length_from_SI(param.showAffineScale)

    return body


class SceneArticulatedRigidScale(SofaPython.sml.BaseScene):
    """ Builds a (sub)scene from a model using compliant formulation
    [tag] solid tagged with rigidScale are simulated as ShearlessAffineBody, more tags can be added to param.rigidScaleTags
    [tag] mesh group tagged with rigidScalePosition are used to compute (barycenter) the positions of a rigidScale
    Compliant joints are setup between the bones """


    def __init__(self, parentNode, model):
        SofaPython.sml.BaseScene.__init__(self, parentNode, model)

        self.rigidScales = dict()
        self.joints = dict()

        ## params

        # the set of tags simulated as rigids
        self.param.rigidScaleTags={"rigidScale"}
        self.param.voxelSize = 0.005 # SI unit (m)
        # simulation
        self.param.rigidScaleStiffness = 10e3 # SI unit
        # for tagged joints, values come from these dictionnaries if they contain one of the tag

        self.param.jointIsComplianceByTag=dict()
        self.param.jointIsComplianceByTag["default"]=False
        self.param.jointComplianceByTag=dict()
        self.param.jointComplianceByTag["default"]=1e-6

        # visual
        self.param.showAffine=False
        self.param.showAffineScale=0.05 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.01 # SI unit (m)

    def createScene(self):
        self.node.createObject('RequiredPlugin', name='image')
        self.node.createObject('RequiredPlugin', name='Flexible')
        self.node.createObject('RequiredPlugin', name='Compliant')
        self.node.createObject('RequiredPlugin', name='RigidScale')

        # rigidScale
        for tag in self.param.rigidScaleTags:
            if tag in self.model.solidsByTag:
                for solidModel in self.model.solidsByTag[tag]:
                    self.rigidScales[solidModel.id] = insertRigidScale(self.node, solidModel, self.param)
        # joints
        for jointModel in self.model.genericJoints.values():
            self.joints[jointModel.id] = Compliant.sml.insertJoint(jointModel, self.rigidScales, self.param)

