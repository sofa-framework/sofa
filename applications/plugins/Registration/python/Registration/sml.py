import Compliant.sml
import Compliant.Tools as Tools
import SofaPython
import os


class SceneRegisterArticulatedRigid(Compliant.sml.SceneArticulatedRigid):
    """
    [bone] are simulated as rigids
    [skin] is mapped under the rigids
    Registration forcefield are added to [bone] and [skin]
    Multiple collision meshes in a rigid isn't supported
    """

    class Target:
        def __init__(self):
            pass

    def __init__(self, parentNode, model):
        Compliant.sml.SceneArticulatedRigid.__init__(self, parentNode, model)

        # Param
        self.param.density_bone = 1500  # kg/m3
        self.param.blendingFactor = 1
        self.param.stiffness = 1
        self.param.outlierThreshold = 0
        self.param.normalThreshold = 0.5
        self.param.rejectOutsideBbox = False
        self.param.damping = 1E-3
        self.param.invertSourceNormals = False

        # target param
        self.param.target = SceneRegisterArticulatedRigid.Target()
        self.param.target.show = True
        self.param.target.scale = "1 1 1"

        # add rigid tag to bones and set densities
        if not self.model.getSolidsByTags({"rigid"}):  # Maybe rigid tags are in the model
            for bone in self.model.getSolidsByTags({"bone"}):
                bone.tags.add("rigid")
                bone.density = self.param.density_bone
            model.updateTag()

    def createScene(self):

        Compliant.sml.SceneArticulatedRigid.createScene(self)
        self.node.createObject('RequiredPlugin', name='Registration')

        # Merge rigids in one node
        self.nodes["dofRigid"] = self.insertMergeRigid("dofRigid")

        # Target node        
        targets = self.model.getSolidsByTags({"target"})
        if targets:
            self.__insert_target(targets)
        else:
            print("<Registration> error: No target detected")
            return

        # Registration node creation
        self.__insert_registration_node()

    def __insert_target(self,targets):
        """ Insert the target node in the graph"""

        if len(targets) != 1:
            print("<Registration> error: multiple target detected")
            return

        # Target properties
        target = targets[0]
        file = target.mesh[0].source
        translation = SofaPython.Tools.listToStr(target.position[:3])
        rotation = SofaPython.Tools.listToStr(target.position[3:])

        # node creation
        target_node = self.createChild(self.node, "Target")
        (filename, ext) = os.path.splitext(os.path.basename(file))

        if ext == ".obj":
            target_node.createObject('MeshObjLoader', name='loader', filename=file, triangulate='1',
                                     translation=translation, rotation=rotation, scale3d=self.param.target.scale)

        if ext == ".stl":
            target_node.createObject('MeshSTLLoader', name='loader', filename=file, triangulate='1',
                                     translation=translation, rotation=rotation, scale3d=self.param.target.scale)

        target_node.createObject('NormalsFromPoints', name='NormalsFromPoints', template='Vec3d', src='@loader',
                                 invertNormals=0)

        # Target visual model
        if self.param.target.show:
            target_node.createObject('VisualModel', name="visual", src="@loader")

    def __insert_registration_node(self):
        """ Insert the registration node in the graph, under dofRigid node. """
        registration_node = self.createChild(self.nodes["dofRigid"], "RegistrationNode")

        param = self.param  # shortcut

        # merge topologies
        repartition = ""
        sources_component_name = []

        collision_node = [elem.collision for elem in self.rigids.values()]

        for i, elem in enumerate(collision_node):
            sources_component_name.append(Tools.node_path_rel(registration_node, elem.node) + "/topology")

            repartition += "{0} ".format(i) * len(elem.topology.position)

        registration_node.createObject('MergeMeshes', name='source_topology', nbMeshes=len(sources_component_name),
                                       **dict({'position' + str(i + 1): '@' + item + '.position' for i, item in
                                               enumerate(sources_component_name)},
                                              **{'triangles' + str(i + 1): '@' + item + '.triangles' for i, item in
                                                 enumerate(sources_component_name)}))

        registration_node.createObject('MeshTopology', name='topo', src='@./source_topology')
        registration_node.createObject('MechanicalObject', name='DOFs')
        registration_node.createObject('Triangle')

        registration_node.createObject('RigidMapping', template='Rigid3d,Vec3d',
                                       input="@" + Tools.node_path_rel(registration_node, self.nodes["dofRigid"]),
                                       output="@DOFs", rigidIndexPerPoint=repartition)

        registration_node.createObject('NormalsFromPoints', name='NormalsFromPoints',
                                       template='Vec3d', position='@DOFs.position',
                                       triangles='@topo.triangles',
                                       invertNormals=self.param.invertSourceNormals)

        # Force Field
        target_path = Tools.node_path_rel(registration_node, self.nodes["Target"])
        registration_node.createObject('ClosestPointRegistrationForceField', name='ICP',
                                       template='Vec3d',
                                       # source
                                       sourceTriangles='@topo.triangles',
                                       sourceNormals='@NormalsFromPoints.normals',
                                       # target
                                       position='@{0}/loader.position'.format(target_path),
                                       triangles='@{0}/loader.triangles'.format(target_path),
                                       normals='@{0}/NormalsFromPoints.normals'.format(target_path),
                                       # Param
                                       cacheSize='4', blendingFactor=param.blendingFactor,
                                       stiffness=param.stiffness,
                                       damping=param.damping,
                                       outlierThreshold=param.outlierThreshold,
                                       normalThreshold=param.normalThreshold,
                                       rejectOutsideBbox=param.rejectOutsideBbox,
                                       drawColorMap='0')


class SceneRegisterArticulatedAffine(Compliant.sml.SceneArticulatedRigid):
    """
    [bone] are simulated as affines
    [skin] is mapped under the affines
    Registration forcefield are added to [bone] and [skin]
    """
    pass
