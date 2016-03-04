import Sofa
import SofaPython.Tools
import json

def exportImageShapeFunction( node, sf, filename_indices, filename_weights ):
    sfPath = SofaPython.Tools.getObjectPath(sf)
    node.createObject(
        "ImageExporter", template="ImageUI", name="exporterIndices",
        image="@"+sfPath+".indices", transform="@"+sfPath+".transform",
        filename=filename_indices,
        exportAtBegin=True, printLog=True)
    node.createObject(
        "ImageExporter", template="ImageR", name="exporterWeights",
        image="@"+sfPath+".weights", transform="@"+sfPath+".transform",
        filename=filename_weights,
        exportAtBegin=True, printLog=True)

def importImageShapeFunction( node, filename_indices, filename_weights, dofname ):
        node.createObject(
            "ImageContainer", template="ImageUI", name="containerIndices",
            filename=filename_indices, drawBB=False)
        node.createObject(
            "ImageContainer", template="ImageR", name="containerWeights",
            filename=filename_weights, drawBB=False)
        return node.createObject(
            "ImageShapeFunctionContainer", template="ShapeFunctiond,ImageUC", name="SF",
            position='@'+dofname+'.rest_position',
            transform="@containerWeights.transform",
            weights="@containerWeights.image", indices="@containerIndices.image")

def exportGaussPoints(gausssampler,filename):
        volumeDim = len(gausssampler.volume)/ len(gausssampler.position) if isinstance(gausssampler.volume, list) is True else 1 # when volume is a list (several GPs or order> 1)
        data = {'volumeDim': str(volumeDim), 'inputVolume': SofaPython.Tools.listListToStr(gausssampler.volume), 'position': SofaPython.Tools.listListToStr(gausssampler.position)}
        with open(filename, 'w') as f:
            json.dump(data, f)
            print "exporting gauss points " + filename

def importGaussPoints(node,filename):
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        return node.createObject('GaussPointContainer',name='sampler', volumeDim=data['volumeDim'], inputVolume=data['inputVolume'], position=data['position'])

def exportLinearMapping(mapping,filename):
        data = {'indices': mapping.indices, 'weights': mapping.weights,
                'weightGradients': mapping.weightGradients, 'weightHessians': mapping.weightHessians}
        with open(filename, 'w') as f:
            json.dump(data, f)
            print "exporting mapping "+SofaPython.Tools.getObjectPath(mapping)+" in "+filename

def importLinearMapping(node,filename):
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        return node.createObject('LinearMapping', indices= str(data['indices']), weights= str(data['weights']), weightGradients= str(data['weightGradients']), weightHessians= str(data['weightHessians']) )

def exportAffineMass(affineMass,filename):
        data = {'affineMassMatrix': affineMass.massMatrix}
        with open(filename, 'w') as f:
            json.dump(data, f)
            print "exporting Affine Mass "+filename

def importAffineMass(node,filename):
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        return node.createObject('AffineMass',name='affineMass', massMatrix=data['affineMassMatrix'])


def exportRigidDofs(dofs,filename):
        str = SofaPython.Tools.listToStr(dofs.position)
        str = str.replace(',',' ')
        str = str.replace('[',' ')
        str = str.replace(']',' ')
        data = {'position': str}
        with open(filename, 'w') as f:
            json.dump(data, f)
            print "exporting Rigid dofs "+filename

def importRigidDofs(node,filename):
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        return node.createObject('MechanicalObject',template="Rigid3d",name='dofs', position=data['position'])
