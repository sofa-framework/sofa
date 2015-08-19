import SofaPython.sml

def getSolidSkinningIndicesAndWeights(solidModel, skinningArmatureBoneIndexById) :
    """ Construct the indices and weights vectors for the skinning of solidModel
    """
    indices = dict()
    weights = dict()
    for skinning in solidModel.skinnings:
        currentBoneIndex = skinningArmatureBoneIndexById[skinning.solid.id]
        for index,weight in zip(skinning.index, skinning.weight):
            if not index in indices:
                indices[index]=list()
                weights[index]=list()
            indices[index].append(currentBoneIndex)
            weights[index].append(weight)
    #TODO fill potential holes in indices/weights ?
    return (indices, weights)
