
import Sofa

def createScene(rootNode):

    rootNode.createObject('RequiredPlugin',pluginName='Flexible')
    rootNode.createObject('RequiredPlugin',pluginName='image')

    rootNode.findData('gravity').value='0 -10 0'
    rootNode.findData('dt').value='1'
    rootNode.createObject('BackgroundSetting',color='1 1 1')
    rootNode.createObject('VisualStyle',displayFlags='showVisual')

    rootNode.createObject('DefaultAnimationLoop')
    rootNode.createObject('DefaultVisualManagerLoop')

    rootNode.createObject('MeshObjLoader',name='loader',filename='mesh/Armadillo_simplified.obj',triangulate='1')
    rootNode.createObject('MeshToImageEngine',name='rasterizer',src='@loader',voxelSize='1.5',padSize='1',rotateImage='false')
    rootNode.createObject('ImageContainer',template='ImageB',name='image',src='@rasterizer',drawBB='false')
    # rootNode.createObject('ImageViewer',template='ImageB',name='viewer',src='@image')
    sampler = rootNode.createObject('ImageSampler',template='ImageB',name='sampler',src='@image',param='1',showEdges='false',printLog='true')

    node = rootNode.createChild('fineVolumes (blue)')

    node.createObject('EulerImplicit',rayleighMass='1',rayleighStiffness='0.03')
    node.createObject('CGLinearSolver',iterations='50',tolerance='1e-5',threshold='1e-5')

    node.createObject('Mesh',name='mesh',src='@../sampler')
    node.createObject('MechanicalObject',name='dofs',tags='NoPicking',position='@./mesh.position')
    node.createObject('UniformMass',totalMass='50')

    node.createObject('BoxROI',template='Vec3d',box='5 7 -8 10 15 3',position='@mesh.position',name='FixedROI',drawBoxes='1')
    node.createObject('FixedConstraint',indices='@FixedROI.indices')

    node.createObject('BarycentricShapeFunction',nbRef='8')

    ############# Fine volume generation using voxel samples
    volnode = node.createChild('Volumes')
    rasterizer = volnode.createObject('MeshToImageEngine',name='rasterizer',src='@../../loader',voxelSize='.5',padSize='1',rotateImage='false')
    volnode.createObject('ImageContainer',template='ImageB',name='image',src='@rasterizer',drawBB='false')
    # volnode.createObject('ImageViewer',template='ImageB',name='viewer',src='@image')
    volnode.createObject('ImageSampler',template='ImageB',name='sampler',src='@image',param='0',showSamplesScale=.0, drawMode=0)
    volnode.createObject('MechanicalObject',name='dof',position='@sampler.position')
    mapping = volnode.createObject('LinearMapping',template='Vec3d,Vec3d',assemble='false') # the linear mapping is used to retrieve hexa containing samples
    rootNode.init() # used to initalize all variables for python functions
    voxelVolume = float(rasterizer.voxelSize[0][0])*float(rasterizer.voxelSize[0][0])*float(rasterizer.voxelSize[0][0])
    fineVolumes= getHexaVolumes(len(sampler.hexahedra),  getHexaIndices(sampler.hexahedra, tohlist(mapping.indices)), voxelVolume)
    #############

    bnode = node.createChild('behavior')
    bnode.createObject('TopologyGaussPointSampler',name='sampler',inPosition='@../dofs.rest_position',method='0',order='2',fineVolumes=' '.join(map(str, fineVolumes)))
    bnode.createObject('MechanicalObject',template='F331',name='F',showObject='0',showObjectScale='0.05')
    bnode.createObject('LinearMapping',template='Vec3d,F331',assemble='false',parallel='false')

    enode = bnode.createChild('Strain')
    enode.createObject('MechanicalObject',template='E331',name='E')
    enode.createObject('CorotationalStrainMapping',template='F331,E331',method='polar',geometricStiffness='false',assemble='false',parallel='false')
    enode.createObject('HookeForceField',template='E331',name='ff',youngModulus='3000',poissonRatio='0.3',viscosity='0')

    cnode = node.createChild('Collision')
    cnode.createObject('Mesh',src='@../../loader')
    cnode.createObject('MechanicalObject',name='dof')
    cnode.createObject('LinearMapping',name='colmap',template='Vec3d,Vec3d',assemble='false',parallel='false')

    vnode = cnode.createChild('Visual')
    vnode.createObject('VisualStyle',displayFlags='hideWireframe')
    vnode.createObject('VisualModel',fileMesh='mesh/Armadillo_simplified.obj',color='blue')
    vnode.createObject('IdentityMapping',template='Vec3d,ExtVec3f')

    hnode = node.createChild('VisuHexa')
    hnode.createObject('VisualStyle',displayFlags='hideWireframe')
    hnode.createObject('VisualModel',color='0.8 0.8 1 0.1')
    hnode.createObject('IdentityMapping')

    hnode = node.createChild('VisuHexa2')
    hnode.createObject('VisualStyle',displayFlags='showWireframe')
    hnode.createObject('VisualModel',color='0.8 0.8 1 1')
    hnode.createObject('IdentityMapping')



    node = rootNode.createChild('coarseVolume (red)')

    node.createObject('EulerImplicit',rayleighMass='1',rayleighStiffness='0.03')
    node.createObject('CGLinearSolver',iterations='50',tolerance='1e-5',threshold='1e-5')

    node.createObject('Mesh',name='mesh',src='@../sampler')
    node.createObject('MechanicalObject',name='dofs',tags='NoPicking',position='@./mesh.position')
    node.createObject('UniformMass',totalMass='50')

    node.createObject('BoxROI',template='Vec3d',box='5 7 -8 10 15 3',position='@mesh.position',name='FixedROI',drawBoxes='1')
    node.createObject('FixedConstraint',indices='@FixedROI.indices')

    node.createObject('BarycentricShapeFunction',nbRef='8')

    bnode = node.createChild('behavior')
    bnode.createObject('TopologyGaussPointSampler',name='sampler',inPosition='@../dofs.rest_position',method='0',order='2')
    bnode.createObject('MechanicalObject',template='F331',name='F',showObject='0',showObjectScale='0.05')
    bnode.createObject('LinearMapping',template='Vec3d,F331',assemble='false',parallel='false')

    enode = bnode.createChild('Strain')
    enode.createObject('MechanicalObject',template='E331',name='E')
    enode.createObject('CorotationalStrainMapping',template='F331,E331',method='polar',geometricStiffness='false',assemble='false',parallel='false')
    enode.createObject('HookeForceField',template='E331',name='ff',youngModulus='3000',poissonRatio='0.3',viscosity='0')

    cnode = node.createChild('Collision')
    cnode.createObject('Mesh',src='@../../loader')
    cnode.createObject('MechanicalObject',name='dof')
    cnode.createObject('LinearMapping',name='colmap',template='Vec3d,Vec3d',assemble='false',parallel='false')

    vnode = cnode.createChild('Visual')
    vnode.createObject('VisualStyle',displayFlags='hideWireframe')
    vnode.createObject('VisualModel',fileMesh='mesh/Armadillo_simplified.obj',color='red')
    vnode.createObject('IdentityMapping',template='Vec3d,ExtVec3f')

    hnode = node.createChild('VisuHexa')
    hnode.createObject('VisualStyle',displayFlags='hideWireframe')
    hnode.createObject('VisualModel',color='0.8 0.8 1 0.1')
    hnode.createObject('IdentityMapping')

    hnode = node.createChild('VisuHexa2')
    hnode.createObject('VisualStyle',displayFlags='showWireframe')
    hnode.createObject('VisualModel',color='0.8 0.8 1 1')
    hnode.createObject('IdentityMapping')

    return rootNode



def tohlist(s):
    """
        convert sofa string to list of hexa
    """
    sl= s.replace('[','').split(']')
    hlist=list()
    for h in sl:
        if len(h)!=0 :
            hlist.append(map(int,h.split(',')))
    return hlist

def getHexaIndices(hexa, hlist):
    """
        find the hexa index in 'hexa' from a list of hexa sotred in hlist
    """
    indices = [-1 for i in xrange(len(hlist))]

    for i,h in enumerate(hlist):
        h.sort()
        for i2,h2 in enumerate(hexa):
            h2.sort()
            if h==h2:
                indices[i]=i2
    return indices

def getHexaVolumes(nb,indices,sampleVol):
    """
        accumulate hexa volumes from samples
    """
    volumes = [0 for i in xrange(nb)]
    for i in indices:
        volumes[i]+=sampleVol
    return volumes
