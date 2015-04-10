import Compliant.sml

class SceneRegisterArticulatedRigid(Compliant.sml.SceneArticulatedRigid):
    """
    [bone] are simulated as rigids
    [skin] is mapped under the rigids
    Registration forcefield are added to [bone] and [skin]
    """
    def __init__(self, parentNode, model, dataDir):
        Compliant.sml.SceneArticulatedRigid.__init__(self, parentNode, model)

        # add rigid tag to bones and  set densities
        for bone in self.model.solidsByTag["bone"]:
            bone.tags.add("rigid")
            bone.density = self.param.density_bone
        model.updateTag()

    def createScene(self):
        Compliant.sml.SceneArticulatedRigid.createScene(self)

        self.dofRigid = self.createChild(self.node, "dofRigid")
        self.insertMergeRigid(self.nodes["dofRigid"], "bone")


class SceneRegisterArticulatedAffine(Compliant.sml.SceneArticulatedRigid):
    """
    [bone] are simulated as affines
    [skin] is mapped under the affines
    Registration forcefield are added to [bone] and [skin]
    """
    pass
