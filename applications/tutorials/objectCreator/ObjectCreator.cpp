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

#include "ObjectCreator.h"

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
#include <sofa/component/collision/TreeCollisionGroupManager.h>
#ifdef SOFA_HAVE_BGL
#include <sofa/component/collision/BglCollisionGroupManager.h>
#endif
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

simulation::Node::SPtr SimpleObjectCreator::CreateRootWithCollisionPipeline(const std::string &simulationType, const std::string& responseType)
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

#ifdef SOFA_HAVE_BGL
    //--> adding component to handle groups of collision.
    if (simulationType == "bgl")
    {
        component::collision::BglCollisionGroupManager::SPtr collisionGroupManager = sofa::core::objectmodel::New<component::collision::BglCollisionGroupManager>();
        collisionGroupManager->setName("Collision Group Manager");
        root->addObject(collisionGroupManager);
    }
    else
#endif
        if (simulationType == "tree")
        {
            //--> adding component to handle groups of collision.
            component::collision::TreeCollisionGroupManager::SPtr collisionGroupManager = sofa::core::objectmodel::New<component::collision::TreeCollisionGroupManager>();
            collisionGroupManager->setName("Collision Group Manager");
            root->addObject(collisionGroupManager);
        }
    return root;
}

simulation::Node::SPtr  SimpleObjectCreator::CreateEulerSolverNode(simulation::Node::SPtr parent, const std::string& name, const std::string &scheme)
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


simulation::Node::SPtr SimpleObjectCreator::CreateObstacle(simulation::Node::SPtr  parent, const std::string &filenameCollision, const std::string filenameVisual,  const std::string& color,
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


simulation::Node::SPtr SimpleObjectCreator::CreateCollisionNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof, const std::string &filename, const std::vector<std::string> &elements,
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

simulation::Node::SPtr SimpleObjectCreator::CreateVisualNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof,  const std::string &filename, const std::string& color,
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



simulation::Node::SPtr SimpleObjectCreator::CreateCollisionNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
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
//    meshTorus_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    meshTorus_surf->setSrc("", loader_surf.get());
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

simulation::Node::SPtr SimpleObjectCreator::CreateVisualNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::string& color,
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


void SimpleObjectCreator::AddCollisionModels(simulation::Node::SPtr CollisionNode, const std::vector<std::string> &elements)
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
simulation::Node::SPtr SimpleObjectCreator::createGridScene(Vec3 startPoint, Vec3 endPoint, unsigned numX, unsigned numY, unsigned numZ, double totalMass, double stiffnessValue, double dampingRatio )
{
    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-1,0) );
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
    MechanicalObjectRigid3d::SPtr rigid_dof = addNew<MechanicalObjectRigid3d>(rigidNode, "dof");
    UniformMassRigid3d::SPtr rigid_mass = addNew<UniformMassRigid3d>(rigidNode,"mass");

    // Particles mapped to the rigid object
    Node::SPtr mappedParticles = rigidNode->createChild("mappedParticles");
    MechanicalObject3d::SPtr mappedParticles_dof = addNew< MechanicalObject3d>(mappedParticles,"dof");
    RigidMappingRigid3d_to_3d::SPtr mappedParticles_mapping = addNew<RigidMappingRigid3d_to_3d>(mappedParticles,"mapping");
    mappedParticles_mapping->setModels( rigid_dof.get(), mappedParticles_dof.get() );

    // The independent particles
    Node::SPtr independentParticles = simulatedScene->createChild("independentParticles");
    MechanicalObject3d::SPtr independentParticles_dof = addNew< MechanicalObject3d>(independentParticles,"dof");

    // The deformable grid, connected to its parents using a MultiMapping
    Node::SPtr deformableGrid = independentParticles->createChild("deformableGrid");

    RegularGridTopology::SPtr deformableGrid_grid = addNew<RegularGridTopology>( deformableGrid, "grid" );
    deformableGrid_grid->setNumVertices(numX,numX,numZ);
    deformableGrid_grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3d::SPtr deformableGrid_dof = addNew< MechanicalObject3d>(deformableGrid,"dof");

    SubsetMultiMapping3d_to_3d::SPtr deformableGrid_mapping = addNew<SubsetMultiMapping3d_to_3d>(deformableGrid,"mapping");
    deformableGrid_mapping->addInputModel(independentParticles_dof.get());
    deformableGrid_mapping->addInputModel(mappedParticles_dof.get());
    deformableGrid_mapping->addOutputModel(deformableGrid_dof.get());

    UniformMass3::SPtr mass = addNew<UniformMass3>(deformableGrid,"mass" );
    mass->mass.setValue( totalMass/(numX*numY*numZ) );

    RegularGridSpringForceField3::SPtr spring = addNew<RegularGridSpringForceField3>(deformableGrid, "spring");
    spring->setLinesStiffness(stiffnessValue);
    spring->setLinesDamping(dampingRatio);


    // ======  Set up the multimapping and its parents, based on its child
    // initialize the grid, so that the particles are located in space
    deformableGrid_grid->init();
    deformableGrid_dof->init();
    cerr<<"size = "<< deformableGrid_dof->getSize() << endl;
    MechanicalObject3::ReadVecCoord  xgrid = deformableGrid_dof->readPositions();
    cerr<<"xgrid = " << xgrid << endl;

    // find the particles attached to the rigid object: x=xMin
    BoxROI3d::SPtr deformableGrid_boxRoi = addNew<BoxROI3d>(deformableGrid,"boxROI");
    deformableGrid_boxRoi->f_X0.setValue(xgrid.ref()); // consider initial positions only
    double eps = (endPoint[0]-startPoint[0])/(numX*2);
    write(deformableGrid_boxRoi->boxes).resize(1);
    write(deformableGrid_boxRoi->boxes).push_back(
       Vec6d(
         startPoint[0]-eps,startPoint[1]-eps,startPoint[2]-eps,
         startPoint[0]+eps,endPoint[1]+eps,endPoint[2]-eps
        )
    ); //  find particles such that x=xMin
    deformableGrid_boxRoi->init();
    helper::vector<unsigned> indices = deformableGrid_boxRoi->f_indices.getValue();
    cerr<<"Indices of the grid in the box: " << indices << endl;
    std::sort(indices.begin(),indices.end());

    // copy each grid particle either in the mapped parent, or in the independent parent, depending on its index
    mappedParticles_dof->resize(indices.size());
    independentParticles_dof->resize( numX*numY*numZ - indices.size() );
    MechanicalObject3::WriteVecCoord xmapped = mappedParticles_dof->writePositions();
    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions();
    assert(indices.size()>0);
    unsigned mappedIndex=0,independentIndex=0;
    for( unsigned i=0; i<xgrid.size(); i++ )
    {
        if( mappedIndex<indices.size() && i==indices[mappedIndex] ){ // mapped particle
            deformableGrid_mapping->addPoint(mappedParticles_dof.get(),i);
            xmapped[mappedIndex] = xgrid[i];
            mappedIndex++;
        }
        else { // independent particle
            deformableGrid_mapping->addPoint(independentParticles_dof.get(),i);
            xindependent[independentIndex] = xgrid[i];
            independentIndex++;
        }
    }

    return root;

}


}
