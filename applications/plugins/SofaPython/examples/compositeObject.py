import Sofa

# a bounding box utility class...
class BoundingBox(object):
    def __init__(self,mini,maxi):
        self.mini=[mini.x,mini.y,mini.z]
        self.maxi=[maxi.x,maxi.y,maxi.z]

    def contains(self,vec):
        if self.mini[0]>vec[0]:
            return False
        if self.mini[1]>vec[1]:
            return False
        if self.mini[2]>vec[2]:
            return False
        if self.maxi[0]<vec[0]:
            return False
        if self.maxi[1]<vec[1]:
            return False
        if self.maxi[2]<vec[2]:
            return False
        return True




# Create an assembly of a siff hexahedral grid with other objects
def createGridScene(rootNode, startPoint, endPoint, numX, numY, numZ, totalMass, stiffnessValue, dampingRatio):


# The graph root node
    rootNode.setGravity( Sofa.Vector3(0,-10,0) )
    rootNode.dt = 0.01
    rootNode.createObject('VisualStyle', displayFlags='hideVisual showBehavior hideCollision showMapping hideOptions')

    simulatedScene = rootNode.createChild('simulatedScene')

    eulerImplicitSolver = simulatedScene.createObject('EulerImplicitSolver')
    cgLinearSolver = simulatedScene.createObject('CGLinearSolver')

# The rigid object
    rigidNode = simulatedScene.createChild('rigidNode')
    rigid_dof = rigidNode.createObject('MechanicalObject', template='Rigid', name='rigidNode_dof')
    rigid_mass = rigidNode.createObject('UniformMass', template='Rigid', name='rigidNode_mass')
    rigid_fixedConstraint = rigidNode.createObject('FixedConstraint', template='Rigid', name='rigidNode_fixedConstraint')

# Particles mapped to the rigid object
    mappedParticles = rigidNode.createChild('mappedParticles')
    mappedParticles_dof = mappedParticles.createObject('MechanicalObject', name='mappedParticles_dof')
    mappedParticles_mapping = mappedParticles.createObject('RigidMapping', template='Rigid', name='mappedParticles_mapping', input='@./../rigidNode_dof', output='@./mappedParticles_dof')

#    // The independent particles
    independentParticles = simulatedScene.createChild('independentParticles')
    independentParticles_dof = independentParticles.createObject('MechanicalObject', name='independentParticles_dof')

#    // The deformable grid, connected to its 2 parents using a MultiMapping
    deformableGrid = independentParticles.createChild('deformableGrid') # first parent
    mappedParticles.addChild(deformableGrid) # second parent

    deformableGrid_grid = deformableGrid.createObject('RegularGridTopology', name='deformableGrid_grid')
    deformableGrid_grid.setNumVertices(numX,numY,numZ)
    deformableGrid_grid.setPos(startPoint.x,endPoint.x,startPoint.y,endPoint.y,startPoint.z,endPoint.z)

    deformableGrid_dof = deformableGrid.createObject('MechanicalObject', name='deformableGrid_dof')
#    deformableGrid_mapping = mappedParticles.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', name='mapping', input='@../../simulatedScene/independentParticles/independentParticles_dof @./mappedParticles_dof', output='@../../simulatedScene/independentParticles/deformableGrid/deformableGrid_dof')
    deformableGrid_mapping = deformableGrid.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', name='mapping', input='@/simulatedScene/independentParticles/independentParticles_dof @/simulatedScene/rigidNode/mappedParticles/mappedParticles_dof', output='@./deformableGrid_dof')
 
    mass = deformableGrid.createObject('UniformMass', template='Vec3d', name='deformableGrid_mass')
    mass.mass = totalMass/(numX*numY*numZ)

    hexaFem = deformableGrid.createObject('HexahedronFEMForceField', template='Vec3d', name='deformableGrid_hexaFEM')
    hexaFem.youngModulus = 1000
    hexaFem.poissonRatio = 0.4

#    // ======  Set up the multimapping and its parents, based on its child
    deformableGrid_grid.init()  # initialize the grid, so that the particles are located in space
    deformableGrid_dof.init()   # create the state vectors
    xgrid = deformableGrid_dof.position #    xgrid = deformableGrid_dof.readPositions()   #    cerr<<"xgrid = " << xgrid << endl;
    # xgrid is not the positions vector of the dof, but a copy of it

#    // create the rigid frames and their bounding boxes
    numRigid = 2
    boxes = []      #    vector<BoundingBox> boxes(numRigid);
    indices = [[],[]]    #    vector< vector<unsigned> > indices(numRigid); // indices of the particles in each box
    eps = (endPoint.x-startPoint.x)/(numX*2.0)

#    // first box, x=xmin
    boxes.append(BoundingBox(Sofa.Vector3(startPoint.x-eps, startPoint.y-eps, startPoint.z-eps),Sofa.Vector3(startPoint.x+eps, endPoint.y+eps, endPoint.z+eps)))

#    // second box, x=xmax
    boxes.append(BoundingBox(Sofa.Vector3(endPoint.x-eps, startPoint.y-eps, startPoint.z-eps),Sofa.Vector3(endPoint.x+eps, endPoint.y+eps, endPoint.z+eps)))

    rigid_dof.resize(numRigid);
#    MechanicalObjectRigid3d::WriteVecCoord xrigid = rigid_dof->writePositions();
    # for xrigid, we create a list of 7-coords lists (position+orientation), then we will copy it in rigid_dof.position when we are done
    xrigid = []
    xrigid.append([startPoint.x, 0.5*(startPoint.y+endPoint.y), 0.5*(startPoint.z+endPoint.z), 0,0,0,1])
    xrigid.append([endPoint.x, 0.5*(startPoint.y+endPoint.y), 0.5*(startPoint.z+endPoint.z), 0,0,0,1])
    rigid_dof.position = xrigid

#    // find the particles in each box
    isFree = [True]*len(xgrid)
    numMapped = 0
    for i in range(len(xgrid)):
        for b in range(numRigid):
            if isFree[i] and boxes[b].contains(xgrid[i]):
                indices[b].append(i)  # associate the particle with the box
                isFree[i] = False
                numMapped = numMapped + 1

#    // distribution of the grid particles to the different parents (independent particle or solids.
    parentParticles = [None]*len(xgrid)  #    vector< pair<MechanicalObject3d*,unsigned> > parentParticles(xgrid.size());

#    // Copy the independent particles to their parent DOF
    independentParticles_dof.resize( numX*numY*numZ - numMapped )
    xindependent = independentParticles_dof.position  #    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions(); // parent positions
    independentIndex = 0
    for i in range(len(xgrid)):
        if isFree[i]:
            parentParticles[i]=[independentParticles_dof,independentIndex]
            xindependent[independentIndex]=xgrid[i]
            independentIndex = independentIndex+1
    independentParticles_dof.position = xindependent

#    // Mapped particles. The RigidMapping requires to cluster the particles based on their parent frame.
    mappedParticles_dof.resize(numMapped)
#    print 'numMapped='+str(numMapped)
    xmapped = mappedParticles_dof.position  #    MechanicalObject3::WriteVecCoord xmapped = mappedParticles_dof->writePositions(); // parent positions
    print 'xmapped size = '+str(len(xmapped))
    mappedParticles_mapping.globalToLocalCoords = True    # to define the mapped positions in world coordinates
    pointsPerFrame = []   #    vector<unsigned>* pointsPerFrame = mappedParticles_mapping->pointsPerFrame.beginEdit(); // to set how many particles are attached to each frame
    mappedIndex = 0
    print 'len(indices)='+str(len(indices))
#    print 'indices='+str(indices)

    rigidIndexPerPoint = [0]* len(xmapped)

    r = 0
    for ind in indices: # every rigids
        print 'len(ind)='+str(len(ind))
        pointsPerFrame.append(len(ind)) # Tell the mapping the number of points associated with this frame. One box per frame

        for i in ind:
            print 'i='+str(i)+'    mappedIndex='+str(mappedIndex)
            parentParticles[i] = [mappedParticles_dof,mappedIndex]
#            print 'xgrid[i]'+str(xgrid[i])
            xmapped[mappedIndex] = xgrid[i]
            rigidIndexPerPoint[mappedIndex] = r
            mappedIndex = mappedIndex+1
        r+=1

    mappedParticles_dof.position = xmapped


    print "rigidIndexPerPoint "
    print rigidIndexPerPoint
    mappedParticles_mapping.rigidIndexPerPoint = rigidIndexPerPoint
    print mappedParticles_mapping.rigidIndexPerPoint

#    // Declare all the particles to the multimapping
    for pp in parentParticles:
        deformableGrid_mapping.addPoint(pp[0],pp[1])
#    for( unsigned i=0; i<xgrid.size(); i++ )
#    {
#        deformableGrid_mapping->addPoint( parentParticles[i].first, parentParticles[i].second );
#    }


# scene creation method
def createScene(rootNode):
    createGridScene(rootNode, Sofa.Vector3(0,0,0), Sofa.Vector3(5,1,1), 6,2,2, 1.0, 100.0, 0.0 )
    return rootNode
