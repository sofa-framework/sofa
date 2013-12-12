/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#define SOFA_SIMPLEOBJECTCREATOR_CPP

#include "SceneCreator.h"

#include <sofa/helper/system/SetDirectory.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>


//Including Solvers
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

//Including components for collision detection
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/MinProximityIntersection.h>

//Including Collision Models
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/CapsuleModel.h>

//Including Visual Models
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/visualmodel/OglModel.h>

namespace sofa
{

using namespace helper;
using helper::vector;
using namespace simulation;
using namespace core::objectmodel;
using namespace component::odesolver;
using namespace component::container;
using namespace component::topology;
using namespace component::collision;
using namespace component::visualmodel;
using namespace component::mapping;
using namespace component::forcefield;

typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

simulation::Node::SPtr SimpleSceneCreator::CreateRootWithCollisionPipeline(const std::string& responseType)
{

    simulation::Node::SPtr root = simulation::getSimulation()->createNewGraph("root");

    //Components for collision management
    //------------------------------------
    //--> adding collision pipeline
    component::collision::DefaultPipeline::SPtr collisionPipeline = sofa::core::objectmodel::New<component::collision::DefaultPipeline>();
    collisionPipeline->setName("Collision Pipeline");
    root->addObject(collisionPipeline);

    //--> adding collision detection system
    component::collision::BruteForceDetection::SPtr detection = sofa::core::objectmodel::New<component::collision::BruteForceDetection>();
    detection->setName("Detection");
    root->addObject(detection);

    //--> adding component to detection intersection of elements
    component::collision::MinProximityIntersection::SPtr detectionProximity = sofa::core::objectmodel::New<component::collision::MinProximityIntersection>();
    detectionProximity->setName("Proximity");
    detectionProximity->setAlarmDistance(0.3);   //warning distance
    detectionProximity->setContactDistance(0.2); //min distance before setting a spring to create a repulsion
    root->addObject(detectionProximity);

    //--> adding contact manager
    component::collision::DefaultContactManager::SPtr contactManager = sofa::core::objectmodel::New<component::collision::DefaultContactManager>();
    contactManager->setName("Contact Manager");
    contactManager->setDefaultResponseType(responseType);
    root->addObject(contactManager);

    //--> adding component to handle groups of collision.
    component::collision::DefaultCollisionGroupManager::SPtr collisionGroupManager = sofa::core::objectmodel::New<component::collision::DefaultCollisionGroupManager>();
    collisionGroupManager->setName("Collision Group Manager");
    root->addObject(collisionGroupManager);

    return root;
}

simulation::Node::SPtr  SimpleSceneCreator::CreateEulerSolverNode(simulation::Node::SPtr parent, const std::string& name, const std::string &scheme)
{
    simulation::Node::SPtr  node = parent->createChild(name.c_str());

    typedef sofa::component::linearsolver::CGLinearSolver< sofa::component::linearsolver::GraphScatteredMatrix, sofa::component::linearsolver::GraphScatteredVector> CGLinearSolverGraph;

    if (scheme == "Implicit")
    {
        component::odesolver::EulerImplicitSolver::SPtr solver = sofa::core::objectmodel::New<component::odesolver::EulerImplicitSolver>();
        CGLinearSolverGraph::SPtr linear = sofa::core::objectmodel::New<CGLinearSolverGraph>();
        solver->setName("Euler Implicit");
        solver->f_rayleighStiffness.setValue(0.01);
        solver->f_rayleighMass.setValue(1);

        solver->setName("Conjugate Gradient");
        linear->f_maxIter.setValue(25); //iteration maxi for the CG
        linear->f_smallDenominatorThreshold.setValue(1e-05);
        linear->f_tolerance.setValue(1e-05);

        node->addObject(solver);
        node->addObject(linear);
    }
    else if (scheme == "Explicit")
    {
        component::odesolver::EulerSolver::SPtr solver = sofa::core::objectmodel::New<component::odesolver::EulerSolver>();
        solver->setName("Euler Explicit");
        node->addObject(solver);
    }
    else
    {
        std::cerr << "Error: " << scheme << " Integration Scheme not recognized" << std::endl;
    }
    return node;
}


simulation::Node::SPtr SimpleSceneCreator::CreateObstacle(simulation::Node::SPtr  parent, const std::string &filenameCollision, const std::string filenameVisual,  const std::string& color,
                                                           const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node::SPtr  nodeFixed = parent->createChild("Fixed");

    sofa::component::loader::MeshObjLoader::SPtr loaderFixed = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loaderFixed->setName("loader");
    loaderFixed->setFilename(sofa::helper::system::DataRepository.getFile(filenameCollision));
    loaderFixed->load();
    nodeFixed->addObject(loaderFixed);

    component::topology::MeshTopology::SPtr meshNodeFixed = sofa::core::objectmodel::New<component::topology::MeshTopology>();
    meshNodeFixed->setSrc("@"+loaderFixed->getName(), loaderFixed.get());
    nodeFixed->addObject(meshNodeFixed);

    MechanicalObject3::SPtr dofFixed = sofa::core::objectmodel::New<MechanicalObject3>(); dofFixed->setName("Fixed Object");
    dofFixed->setSrc("@"+loaderFixed->getName(), loaderFixed.get());
    dofFixed->setTranslation(translation[0],translation[1],translation[2]);
    dofFixed->setRotation(rotation[0],rotation[1],rotation[2]);
    nodeFixed->addObject(dofFixed);

    component::collision::TriangleModel::SPtr triangleFixed = sofa::core::objectmodel::New<component::collision::TriangleModel>(); triangleFixed->setName("Collision Fixed");
    triangleFixed->setSimulated(false); //Not simulated, fixed object
    triangleFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(triangleFixed);
    component::collision::LineModel::SPtr LineFixed = sofa::core::objectmodel::New<component::collision::LineModel>(); LineFixed->setName("Collision Fixed");
    LineFixed->setSimulated(false); //Not simulated, fixed object
    LineFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(LineFixed);
    component::collision::PointModel::SPtr PointFixed = sofa::core::objectmodel::New<component::collision::PointModel>(); PointFixed->setName("Collision Fixed");
    PointFixed->setSimulated(false); //Not simulated, fixed object
    PointFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(PointFixed);

    component::visualmodel::OglModel::SPtr visualFixed = sofa::core::objectmodel::New<component::visualmodel::OglModel>();
    visualFixed->setName("visual");
    visualFixed->setFilename(sofa::helper::system::DataRepository.getFile(filenameVisual));
    visualFixed->setColor(color);
    visualFixed->setTranslation(translation[0],translation[1],translation[2]);
    visualFixed->setRotation(rotation[0],rotation[1],rotation[2]);
    nodeFixed->addObject(visualFixed);
    return nodeFixed;
}


simulation::Node::SPtr SimpleSceneCreator::CreateCollisionNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof, const std::string &filename, const std::vector<std::string> &elements,
                                                                    const Deriv3& translation, const Deriv3 &rotation)
{
    //Node COLLISION
    simulation::Node::SPtr  CollisionNode = parent->createChild("Collision");

    sofa::component::loader::MeshObjLoader::SPtr loader_surf = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loader_surf->setName("loader");
    loader_surf->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    loader_surf->load();
    CollisionNode->addObject(loader_surf);

    component::topology::MeshTopology::SPtr meshTorus_surf= sofa::core::objectmodel::New<component::topology::MeshTopology>();
    meshTorus_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = sofa::core::objectmodel::New<MechanicalObject3>();  dof_surf->setName("Collision Object ");
    dof_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    AddCollisionModels(CollisionNode, elements);

    BarycentricMapping3_to_3::SPtr mechaMapping = sofa::core::objectmodel::New<BarycentricMapping3_to_3>();
    mechaMapping->setModels(dof.get(), dof_surf.get());
    mechaMapping->setPathInputObject("@..");
    mechaMapping->setPathOutputObject("@.");
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

simulation::Node::SPtr SimpleSceneCreator::CreateVisualNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof,  const std::string &filename, const std::string& color,
                                                                 const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node::SPtr  VisualNode =parent->createChild("Visu");

    const std::string nameVisual="Visual";
    const std::string refVisual = "@" + nameVisual;
    const std::string refDof = "@..";// + dof->getName();
    component::visualmodel::OglModel::SPtr visual = sofa::core::objectmodel::New<component::visualmodel::OglModel>();
    visual->setName(nameVisual);
    visual->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    visual->setColor(color.c_str());
    visual->setTranslation(translation[0],translation[1],translation[2]);
    visual->setRotation(rotation[0],rotation[1],rotation[2]);
    VisualNode->addObject(visual);

    BarycentricMapping3_to_Ext3::SPtr mapping = sofa::core::objectmodel::New<BarycentricMapping3_to_Ext3>();
    mapping->setModels(dof.get(), visual.get());
    mapping->setName("Mapping Visual");
    mapping->setPathInputObject(refDof);
    mapping->setPathOutputObject(refVisual);
    VisualNode->addObject(mapping);

    return VisualNode;
}



simulation::Node::SPtr SimpleSceneCreator::CreateCollisionNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
                                                                     const Deriv3& translation, const Deriv3 &rotation)
{
    const std::string refdofRigid = "@../" + dofRigid->getName();
    const std::string dofSurfName = "CollisionObject";
    const std::string refdofSurf = "@"+dofSurfName;
    //Node COLLISION
    simulation::Node::SPtr  CollisionNode =parent->createChild("Collision");


    sofa::component::loader::MeshObjLoader::SPtr loader_surf = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loader_surf->setName("loader");
    loader_surf->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    loader_surf->load();
    CollisionNode->addObject(loader_surf);

    component::topology::MeshTopology::SPtr meshTorus_surf= sofa::core::objectmodel::New<component::topology::MeshTopology>();
    meshTorus_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = sofa::core::objectmodel::New<MechanicalObject3>(); dof_surf->setName(dofSurfName);
    //    dof_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    AddCollisionModels(CollisionNode, elements);

    RigidMappingRigid3_to_3::SPtr mechaMapping = sofa::core::objectmodel::New<RigidMappingRigid3_to_3>();
    mechaMapping->setModels(dofRigid.get(), dof_surf.get());
    mechaMapping->setPathInputObject(refdofRigid);
    mechaMapping->setPathOutputObject(refdofSurf);
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

simulation::Node::SPtr SimpleSceneCreator::CreateVisualNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::string& color,
                                                                  const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node::SPtr  RigidVisualNode =parent->createChild("Visu");

    const std::string nameVisual="Visual";
    const std::string refVisual="@"+nameVisual;
    const std::string refdofRigid="@../"+dofRigid->getName();
    component::visualmodel::OglModel::SPtr visualRigid = sofa::core::objectmodel::New<component::visualmodel::OglModel>();
    visualRigid->setName(nameVisual);
    visualRigid->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    visualRigid->setColor(color);
    visualRigid->setTranslation(translation[0],translation[1],translation[2]);
    visualRigid->setRotation(rotation[0],rotation[1],rotation[2]);
    RigidVisualNode->addObject(visualRigid);

    RigidMappingRigid3_to_Ext3::SPtr mappingRigid = sofa::core::objectmodel::New<RigidMappingRigid3_to_Ext3>();
    mappingRigid->setModels(dofRigid.get(), visualRigid.get());
    mappingRigid->setName("Mapping Visual");
    mappingRigid->setPathInputObject(refdofRigid);
    mappingRigid->setPathOutputObject(refVisual);
    RigidVisualNode->addObject(mappingRigid);
    return RigidVisualNode;
}


void SimpleSceneCreator::AddCollisionModels(simulation::Node::SPtr CollisionNode, const std::vector<std::string> &elements)
{
    for (unsigned int i=0; i<elements.size(); ++i)
    {
        if (elements[i] == "Triangle")
        {
            component::collision::TriangleModel::SPtr triangle = sofa::core::objectmodel::New<component::collision::TriangleModel>();  triangle->setName("TriangleCollision");
            CollisionNode->addObject(triangle);
        }
        else if(elements[i] == "Line")
        {
            component::collision::LineModel::SPtr line = sofa::core::objectmodel::New<component::collision::LineModel>();  line->setName("LineCollision");
            CollisionNode->addObject(line);
        }
        else if (elements[i] == "Point")
        {
            component::collision::PointModel::SPtr point = sofa::core::objectmodel::New<component::collision::PointModel>();  point->setName("PointCollision");
            CollisionNode->addObject(point);
        }
        else if (elements[i] == "Sphere")
        {
            component::collision::SphereModel::SPtr point = sofa::core::objectmodel::New<component::collision::SphereModel>();  point->setName("SphereCollision");
            CollisionNode->addObject(point);
        }
        else if(elements[i] == "Capsule"){
            component::collision::CapsuleModel::SPtr capsule = sofa::core::objectmodel::New<component::collision::CapsuleModel>();  capsule->setName("CapsuleCollision");
            CollisionNode->addObject(capsule);
        }
        else if(elements[i] == "OBB"){
            component::collision::OBBModel::SPtr obb = sofa::core::objectmodel::New<component::collision::OBBModel>();  obb->setName("OBBCollision");
            CollisionNode->addObject(obb);
        }
    }
}

template<class Component>
typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="")
{
    typename Component::SPtr component = New<Component>();
    parentNode->addObject(component);
    component->setName(parentNode->getName()+"_"+name);
    return component;
}


/// Create an assembly of a siff hexahedral grid with other objects
simulation::Node::SPtr SimpleSceneCreator::createGridScene(Vec3 startPoint, Vec3 endPoint, unsigned numX, unsigned numY, unsigned numZ, double totalMass, double stiffnessValue, double dampingRatio )
{
    using helper::vector;

    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-10,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    Node::SPtr simulatedScene = root->createChild("simulatedScene");

    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
    simulatedScene->addObject( eulerImplicitSolver );
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
    simulatedScene->addObject(cgLinearSolver);

    // The rigid object
    Node::SPtr rigidNode = simulatedScene->createChild("rigidNode");
    MechanicalObjectRigid3::SPtr rigid_dof = addNew<MechanicalObjectRigid3>(rigidNode, "dof");
    UniformMassRigid3::SPtr rigid_mass = addNew<UniformMassRigid3>(rigidNode,"mass");
    FixedConstraintRigid3::SPtr rigid_fixedConstraint = addNew<FixedConstraintRigid3>(rigidNode,"fixedConstraint");

    // Particles mapped to the rigid object
    Node::SPtr mappedParticles = rigidNode->createChild("mappedParticles");
    MechanicalObject3::SPtr mappedParticles_dof = addNew< MechanicalObject3>(mappedParticles,"dof");
    RigidMappingRigid3_to_3::SPtr mappedParticles_mapping = addNew<RigidMappingRigid3_to_3>(mappedParticles,"mapping");
    mappedParticles_mapping->setModels( rigid_dof.get(), mappedParticles_dof.get() );

    // The independent particles
    Node::SPtr independentParticles = simulatedScene->createChild("independentParticles");
    MechanicalObject3::SPtr independentParticles_dof = addNew< MechanicalObject3>(independentParticles,"dof");

    // The deformable grid, connected to its 2 parents using a MultiMapping
    Node::SPtr deformableGrid = independentParticles->createChild("deformableGrid"); // first parent
    mappedParticles->addChild(deformableGrid);                                       // second parent

    RegularGridTopology::SPtr deformableGrid_grid = addNew<RegularGridTopology>( deformableGrid, "grid" );
    deformableGrid_grid->setNumVertices(numX,numY,numZ);
    deformableGrid_grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3::SPtr deformableGrid_dof = addNew< MechanicalObject3>(deformableGrid,"dof");

    SubsetMultiMapping3_to_3::SPtr deformableGrid_mapping = addNew<SubsetMultiMapping3_to_3>(deformableGrid,"mapping");
    deformableGrid_mapping->addInputModel(independentParticles_dof.get()); // first parent
    deformableGrid_mapping->addInputModel(mappedParticles_dof.get());      // second parent
    deformableGrid_mapping->addOutputModel(deformableGrid_dof.get());

    UniformMass3::SPtr mass = addNew<UniformMass3>(deformableGrid,"mass" );
    mass->mass.setValue( totalMass/(numX*numY*numZ) );

    RegularGridSpringForceField3::SPtr spring = addNew<RegularGridSpringForceField3>(deformableGrid, "spring");
    spring->setLinesStiffness(stiffnessValue);
    spring->setQuadsStiffness(stiffnessValue);
    spring->setCubesStiffness(stiffnessValue);
    spring->setLinesDamping(dampingRatio);


    // ======  Set up the multimapping and its parents, based on its child
    // initialize the grid, so that the particles are located in space
    deformableGrid_grid->init();
    deformableGrid_dof->init();
    //    cerr<<"SimpleSceneCreator::createGridScene size = "<< deformableGrid_dof->getSize() << endl;
    MechanicalObject3::ReadVecCoord  xgrid = deformableGrid_dof->readPositions();
    //    cerr<<"SimpleSceneCreator::createGridScene xgrid = " << xgrid << endl;


    // create the rigid frames and their bounding boxes
    unsigned numRigid = 2;
    vector<BoundingBox> boxes(numRigid);
    vector< vector<unsigned> > indices(numRigid); // indices of the particles in each box
    double eps = (endPoint[0]-startPoint[0])/(numX*2);

    // first box, x=xmin
    boxes[0] = BoundingBox(Vec3d(startPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           Vec3d(startPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));

    // second box, x=xmax
    boxes[1] = BoundingBox(Vec3d(endPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           Vec3d(endPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));
    rigid_dof->resize(numRigid);
    MechanicalObjectRigid3::WriteVecCoord xrigid = rigid_dof->writePositions();
    xrigid[0].getCenter()=Vec3(startPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));
    xrigid[1].getCenter()=Vec3(  endPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));

    // find the particles in each box
    vector<bool> isFree(xgrid.size(),true);
    unsigned numMapped = 0;
    for(unsigned i=0; i<xgrid.size(); i++){
        for(unsigned b=0; b<numRigid; b++ )
        {
            if( isFree[i] && boxes[b].contains(xgrid[i]) )
            {
                indices[b].push_back(i); // associate the particle with the box
                isFree[i] = false;
                numMapped++;
            }
        }
    }

    // distribute the particles to the different solids. One solid for each box.
    mappedParticles_dof->resize(numMapped);
    independentParticles_dof->resize( numX*numY*numZ - numMapped );
    MechanicalObject3::WriteVecCoord xmapped = mappedParticles_dof->writePositions();
    mappedParticles_mapping->globalToLocalCoords.setValue(true); // to define the mapped positions in world coordinates
    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions();
    vector< pair<MechanicalObject3*,unsigned> > parentParticles(xgrid.size());

    // independent particles
    unsigned independentIndex=0;
    for( unsigned i=0; i<xgrid.size(); i++ ){
        if( isFree[i] ){
            parentParticles[i]=make_pair(independentParticles_dof.get(),independentIndex);
            xindependent[independentIndex] = xgrid[i];
            independentIndex++;
        }
    }

    // mapped particles
    unsigned mappedIndex=0;
    vector<unsigned>* pointsPerFrame = mappedParticles_mapping->pointsPerFrame.beginEdit();
    for( unsigned b=0; b<numRigid; b++ )
    {
        const vector<unsigned>& ind = indices[b];
        pointsPerFrame->push_back(ind.size()); // tell the mapping the number of points associated with this frame
        for(unsigned i=0; i<ind.size(); i++)
        {
            parentParticles[ind[i]]=make_pair(mappedParticles_dof.get(),mappedIndex);
            xmapped[mappedIndex] = xgrid[ ind[i] ];
            mappedIndex++;

        }
    }
    mappedParticles_mapping->pointsPerFrame.endEdit();

    // now add all the particles to the multimapping
    for( unsigned i=0; i<xgrid.size(); i++ )
    {
        deformableGrid_mapping->addPoint( parentParticles[i].first, parentParticles[i].second );
    }


    return root;

}

namespace modeling {

simulation::Node::SPtr newRoot()
{
    return simulation::getSimulation()->createNewGraph("root");
}


/// Create a stiff string
simulation::Node::SPtr massSpringString
(
        simulation::Node::SPtr parent,
        double x0, double y0, double z0, // start point,
        double x1, double y1, double z1, // end point
        unsigned numParticles,
        double totalMass,
        double stiffnessValue,
        double dampingRatio
        )
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "string_" << numObject++;

    Vec3d startPoint(x0,y0,z0), endPoint(x1,y1,z1);
    SReal totalLength = (endPoint-startPoint).norm();

    //--------
    Node::SPtr  string_node = parent->createChild(oss.str());

    MechanicalObject3::SPtr DOF = New<MechanicalObject3>();
    string_node->addObject(DOF);
    DOF->setName(oss.str()+"_DOF");

    UniformMass3::SPtr mass = New<UniformMass3>();
    string_node->addObject(mass);
    mass->setName(oss.str()+"_mass");
    mass->mass.setValue( totalMass/numParticles );

    StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>();
    string_node->addObject(spring);
    spring->setName(oss.str()+"_spring");



    //--------
    // create the particles and the springs
    DOF->resize(numParticles);
    MechanicalObject3::WriteVecCoord x = DOF->writePositions();
    for( unsigned i=0; i<numParticles; i++ )
    {
        double alpha = (double)i/(numParticles-1);
        x[i] = startPoint * (1-alpha)  +  endPoint * alpha;
        if(i>0)
        {
            spring->addSpring(i-1,i,stiffnessValue,dampingRatio,totalLength/(numParticles-1));
         }
    }

    return string_node;

}

}

}
