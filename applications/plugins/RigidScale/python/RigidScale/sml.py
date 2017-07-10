import Sofa
import RigidScale.API
import SofaPython.sml
import Compliant.StructuralAPI
import Compliant.sml
import Flexible
import os

from Compliant.Tools import cat as concat

printLog = True


def insertRigidScale(parentNode, rigidModel, param):
    """
    Create a RigidScale.API.ShearlessAffineBody from the solidModel
    """
    if printLog:
        Sofa.msg_info("RigidScale.sml", "insertRigidScale "+rigidModel.name)

    body = RigidScale.API.ShearlessAffineBody(parentNode, rigidModel.name)

    if (not len(rigidModel.mesh)==1):
        Sofa.msg_warning("RigidScale.sml", "insertRigidScale support only single mesh solid (nb meshes={0}) - solid {1} ignored".format(len(rigidModel.mesh), rigidModel.name))
        return None

    # TODO support multi meshes
    meshFormatSupported = True
    for mesh in rigidModel.mesh :
        meshFormatSupported &= mesh.format=="obj" or mesh.format=="vtk"

    if len(rigidModel.mesh)>0 and meshFormatSupported:
        body.setFromMesh(rigidModel.mesh[0].source,
                     numberOfPoints = SofaPython.sml.getValueByTag(param.rigidScaleNbDofByTag, rigidModel.tags),
                     voxelSize = SofaPython.units.length_from_SI(param.voxelSize),
                     density = SofaPython.units.massDensity_from_SI(1000.),
                     offset = rigidModel.position)
        cm = body.addCollisionMesh(rigidModel.mesh[0].source)
        cm.addVisualModel()
    else:
        body.setManually(offset=rigidModel.position);

    body.addBehavior(youngModulus=SofaPython.units.elasticity_from_SI(param.rigidScaleStiffness), numberOfGaussPoint=8)
    #cm = body.addCollisionMesh(rigidModel.mesh[0].source)
    #cm.addVisualModel()

    body.affineDofs.showObject = param.showAffine
    body.affineDofs.showObjectScale = SofaPython.units.length_from_SI(param.showAffineScale)
    body.rigidDofs.showObject = param.showRigid
    body.rigidDofs.showObjectScale = SofaPython.units.length_from_SI(param.showRigidScale)

    if param.showImage:
        body.image.addViewer()

    return body

def getSolidSkinningIndicesAndWeights(solidModel, meshId, skinningArmatureBoneIndexById) :
    """ Construct the indices and weights vectors for the skinning of solidModel
    """
    indices = dict()
    weights = dict()
    for skinning in solidModel.skinnings:
        if skinning.mesh.id == meshId:
            currentBoneIndex = skinningArmatureBoneIndexById[skinning.solid.id]
            for index,weight in zip(skinning.index, skinning.weight):
                if not index in indices:
                    indices[index]=list()
                    weights[index]=list()
                indices[index].append(currentBoneIndex)
                weights[index].append(weight)
    #TODO fill potential holes in indices/weights ?
    return (indices, weights)

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
        self.param.jointIsComplianceByTag = dict()
        self.param.jointIsComplianceByTag["default"] = False
        self.param.jointComplianceByTag = dict()
        self.param.jointComplianceByTag["default"] = 1e-6
        self.param.rigidScaleNbDofByTag = dict()
        self.param.rigidScaleNbDofByTag["default"] = 1

        # visual
        self.param.showRigid = False
        self.param.showRigidScale = 0.05  # SI unit (m)

        self.param.showAffine = False
        self.param.showAffineScale = 0.05  # SI unit (m)

        self.param.showOffset = False
        self.param.showOffsetScale = 0.01  # SI unit (m)

        self.param.showImage = False

    def createScene(self):
        self.node.createObject('RequiredPlugin', name='image')
        self.node.createObject('RequiredPlugin', name='Flexible')
        self.node.createObject('RequiredPlugin', name='Compliant')
        self.node.createObject('RequiredPlugin', name='RigidScale')

        # rigidScale
        for solidModel in self.model.getSolidsByTags(self.param.rigidScaleTags):
            self.rigidScales[solidModel.id] = insertRigidScale(self.node, solidModel, self.param)
        # joints
        for jointModel in self.model.genericJoints.values():
            self.joints[jointModel.id] = Compliant.sml.insertJoint(jointModel, self.rigidScales, param=self.param)


class SceneSkinningRigidScale(SofaPython.sml.BaseScene):
    """ Builds a (sub)scene from a model using compliant formulation
    [tag] solid tagged with rigidScale are simulated as ShearlessAffineBody, more tags can be added to param.rigidScaleTags
    [tag] mesh group tagged with rigidScalePosition are used to compute (barycenter) the positions of a rigidScale
    Compliant joints are setup between the bones """

    def insertMergeAffine(self, mergeNodeName="dofAffine", tags=None, affineIndexById=None):
        """ Merge all the affine in a single MechanicalObject using a SubsetMultiMapping
        optionnaly give a list of tags to select the rigids which are merged
        return the created node"""
        mergeNode = None
        currentRigidIndex = 0
        input = ""
        indexPairs = ""

        if tags is None:
            _tags = self.param.rigidTags
        else:
            _tags = tags

        for solid in self.model.getSolidsByTags(_tags):
            if not solid.id in self.rigidScales:
                Sofa.msg_warning("RigidScale.sml","SceneArticulatedRigid.insertMergeRigid: "+solid.name+" is not a rigid")
                continue
            rigidScale = self.rigidScales[solid.id]
            if mergeNode is None:
                mergeNode = rigidScale.affineNode.createChild(mergeNodeName)
            else:
                rigidScale.affineNode.addChild(mergeNode)
            input += '@'+rigidScale.affineNode.getPathName()+" "
            indexPairs += str(currentRigidIndex) + " 0 "
            if not affineIndexById is None:
                affineIndexById[solid.id]=currentRigidIndex
            currentRigidIndex += 1
        if input:
            mergeNode.createObject("MechanicalObject", template = "Affine", name="dofs")
            mergeNode.createObject('SubsetMultiMapping', template = "Affine,Affine", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
        else:
            Sofa.msg_warning("Compliant.sml", "insertMergeRigid: no rigid merged")
        return mergeNode

    def __init__(self, parentNode, model):
        SofaPython.sml.BaseScene.__init__(self, parentNode, model)

        self.rigidScales = dict()
        self.joints = dict()
        self.collisions = dict()
        self.visuals = dict()

        ## params
        # the set of tags simulated as rigids
        # simulation
        self.param.rigidScaleStiffness = 10e3 # SI unit
        self.param.mass = 1
        self.param.uniformScale = False
        # for tagged joints, values come from these dictionnaries if they contain one of the tag
        self.param.jointIsComplianceByTag=dict()
        self.param.jointIsComplianceByTag["default"]=False
        self.param.jointComplianceByTag=dict()
        self.param.jointComplianceByTag["default"]=1e-6
        self.param.rigidScaleNbDofByTag=dict()
        self.param.rigidScaleNbDofByTag["default"]=1
        self.skinningArmatureBoneIndexById = dict()
        self.deformables = dict()
        self.skins = dict()

        self.param.attachedSolids = dict() # for visual mapping

        # visual
        self.param.showRigid = True
        self.param.showRigidScale = 0.05  # SI unit (m)
        self.param.showAffine = False
        self.param.showAffineScale = 0.3  # SI unit (m)
        self.param.showOffset = False
        self.param.showOffsetScale = 0.1  # SI unit (m)

    def createScene(self):

        self.node.createObject('RequiredPlugin', name='image')
        self.node.createObject('RequiredPlugin', name='Flexible')
        self.node.createObject('RequiredPlugin', name='Compliant')
        self.node.createObject('RequiredPlugin', name='RigidScale')

        # rigidScale
        for rigidModel in self.model.getSolidsByTags({"armature"}):

            # the global uniformScale specified by default can be overridden
            uniformScale=self.param.uniformScale
            if "scale1" in rigidModel.tags: uniformScale = True
            elif "scale3" in rigidModel.tags: uniformScale = False

            body = RigidScale.API.ShearlessAffineBody(self.node, rigidModel.name)
            body.setManually(offset=[rigidModel.position], mass=0, inertia=[0,0,0], uniformScale=uniformScale)
            body.affineDofs.showObject = self.param.showAffine
            body.affineDofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showAffineScale)
            body.rigidDofs.showObject = self.param.showRigid
            body.rigidDofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showRigidScale)
            self.rigidScales[rigidModel.id] = body

        # joints
        for jointModel in self.model.genericJoints.values():
            self.joints[jointModel.id] = Compliant.sml.insertJoint(jointModel, self.rigidScales, param=self.param)

        # insert node containing all bones of the armature
        self.nodes["armature"] = self.insertMergeAffine(mergeNodeName="armature", tags={"armature"}, affineIndexById=self.skinningArmatureBoneIndexById)
        for solidModel in self.model.solids.values():

            self.visuals[solidModel.id] = {}
            self.collisions[solidModel.id] = {}

            if len(solidModel.skinnings) > 0:  # ignore solid if it has no skinning
                # for each mesh create a Flexible.API.Deformable
                for mesh in solidModel.mesh:
                    # take care only of visual meshes with skinning

                    #TODO better handle visual / simulated / collision
                    if solidModel.meshAttributes[mesh.id].visual:
                        (indices, weights) = getSolidSkinningIndicesAndWeights(solidModel, mesh.id, self.skinningArmatureBoneIndexById)
                        deformable = Flexible.API.Deformable(self.nodes["armature"], solidModel.name+"_"+mesh.name)
                        if solidModel.meshAttributes[mesh.id].simulation:
                            deformable.loadMesh(mesh.source)
                            deformable.addMechanicalObject()
                            deformable.addMass(self.param.mass)
                            deformable.addSkinning(self.nodes["armature"], indices.values(), weights.values())
                            visualDeformable = deformable.addVisual()
                            self.visuals[solidModel.id][mesh.id] = visualDeformable
                            self.skins[mesh.id] = visualDeformable.visual
                            self.collisions[solidModel.id][mesh.id] = deformable
                        else:
                            deformable.loadVisual(mesh.source, initRestPositions = True)
                            deformable.addSkinning(self.nodes["armature"], indices.values(), weights.values(), isMechanical = False)
                            self.visuals[solidModel.id][mesh.id] = deformable
                        self.deformables[mesh.id] = deformable
            else:
                if solidModel.id in self.param.attachedSolids:
                    for mesh in solidModel.mesh:
                        # TODO handle collision vs visual meshes
                        deformable = Flexible.API.Deformable(self.rigidScales[self.param.attachedSolids[solidModel.id]].rigidNode, solidModel.name+"_"+mesh.name)
                        deformable.loadVisual(mesh.source, initRestPositions = True)
                        deformable.node.createObject("RigidMapping", mapForces=False, mapConstraints=False, mapMasses=False, globalToLocalCoords=True)
                        self.visuals[solidModel.id][mesh.id] = deformable




    def addMeshExporters(self, dir, ExportAtEnd=False):
        """ add obj Exporters for each visual model of the scene
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        for k,visuals in self.visuals.iteritems():
            for mid,visual in visuals.iteritems():
                filename = os.path.join(dir, os.path.basename(self.model.meshes[mid].source))
                e = visual.node.createObject('ObjExporter', name='objExporter', filename=filename, printLog=True, exportAtEnd=ExportAtEnd)
                self.meshExporters.append(e)
