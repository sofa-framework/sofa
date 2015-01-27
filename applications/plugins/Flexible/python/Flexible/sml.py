from SofaPython.Tools import listToStr as concat
import SofaPython.sml
import StructuralAPI

def insertDeformableWithSkinning(parentNode, deformableModel, bonesPath, bonesId):
    print "deformable:", deformableModel.name
    deformable=StructuralAPI.Deformable(parentNode, deformableModel.name)
    deformable.setMesh(deformableModel.position, deformableModel.mesh.source)
    deformable.addVisual()
    
    if len(deformableModel.skinnings)>0:
        # build the sofa indices and weights
        indices = dict()
        weights = dict()
        for s in deformableModel.skinnings:
            currentBoneIndex = bonesId.index(s.rigid.id)
            for index,weight in zip(s.index, s.weight):
                if not index in indices:
                    indices[index]=list()
                    weights[index]=list()
                indices[index].append(currentBoneIndex)
                weights[index].append(weight)
        #TODO fill potential holes in indices/weights ?
        deformable.addSkinning(bonesPath, indices.values(), weights.values())
    return deformable

