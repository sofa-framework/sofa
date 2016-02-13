/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
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
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include "GetVectorVisitor.h"
#include "GetAssembledSizeVisitor.h"

//Including Solvers and linear algebra
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaLoader/MeshObjLoader.h>

//Including components for collision detection
#include <SofaBaseCollision/DefaultPipeline.h>
#include <SofaBaseCollision/DefaultContactManager.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <SofaBaseCollision/MinProximityIntersection.h>

//Including Collision Models
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/CapsuleModel.h>

//Including Visual Models
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaOpenglVisual/OglModel.h>

#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaDeformable/StiffSpringForceField.h>

namespace sofa
{
namespace modeling {

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

typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;
typedef component::mapping::BarycentricMapping<defaulttype::Vec3Types, defaulttype::Vec3Types > BarycentricMapping3_to_3;
typedef component::mapping::BarycentricMapping<defaulttype::Vec3Types, defaulttype::ExtVec3fTypes> BarycentricMapping3_to_Ext3;
typedef component::mapping::RigidMapping<defaulttype::Rigid3Types, defaulttype::Vec3Types > RigidMappingRigid3_to_3;
typedef component::mapping::RigidMapping<defaulttype::Rigid3Types, defaulttype::ExtVec3fTypes > RigidMappingRigid3_to_Ext3;
typedef component::mass::UniformMass<defaulttype::Vec3Types, SReal> UniformMass3;
typedef component::interactionforcefield::StiffSpringForceField<defaulttype::Vec3Types > StiffSpringForceField3;

simulation::Node::SPtr createRootWithCollisionPipeline(const std::string& responseType)
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

simulation::Node::SPtr  createEulerSolverNode(simulation::Node::SPtr parent, const std::string& name, const std::string &scheme)
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


simulation::Node::SPtr createObstacle(simulation::Node::SPtr  parent, const std::string &filenameCollision, const std::string filenameVisual,  const std::string& color,
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


simulation::Node::SPtr createCollisionNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof, const std::string &filename, const std::vector<std::string> &elements,
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

    addCollisionModels(CollisionNode, elements);

    BarycentricMapping3_to_3::SPtr mechaMapping = sofa::core::objectmodel::New<BarycentricMapping3_to_3>();
    mechaMapping->setModels(dof.get(), dof_surf.get());
    mechaMapping->setPathInputObject("@..");
    mechaMapping->setPathOutputObject("@.");
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

simulation::Node::SPtr createVisualNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof,  const std::string &filename, const std::string& color,
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



simulation::Node::SPtr createCollisionNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
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

    addCollisionModels(CollisionNode, elements);

    RigidMappingRigid3_to_3::SPtr mechaMapping = sofa::core::objectmodel::New<RigidMappingRigid3_to_3>();
    mechaMapping->setModels(dofRigid.get(), dof_surf.get());
    mechaMapping->setPathInputObject(refdofRigid);
    mechaMapping->setPathOutputObject(refdofSurf);
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

simulation::Node::SPtr createVisualNodeRigid(simulation::Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::string& color,
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


void addCollisionModels(simulation::Node::SPtr CollisionNode, const std::vector<std::string> &elements)
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

//template<class Component>
//typename Component::SPtr addNew( Node::SPtr parentNode, std::string name="")
//{
//    typename Component::SPtr component = New<Component>();
//    parentNode->addObject(component);
//    component->setName(parentNode->getName()+"_"+name);
//    return component;
//}



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

    sofa::defaulttype::Vec3d startPoint(x0,y0,z0), endPoint(x1,y1,z1);
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

Node::SPtr initSofa()
{
    setSimulation(new simulation::graph::DAGSimulation());
    return simulation::getSimulation()->createNewGraph("root");
//    root = modeling::newRoot();
//    root->setName("Solver_test_scene_root");
}


void initScene()
{
    sofa::simulation::getSimulation()->init(getRoot().get());

}

simulation::Node::SPtr clearScene()
{
    if( getRoot() )
        Simulation::theSimulation->unload( getRoot() );
    Simulation::theSimulation->createNewGraph("");
    return getRoot();
}


Node::SPtr getRoot() { return simulation::getSimulation()->GetRoot(); }

Vector getVector( core::ConstVecId id, bool indep )
{
    GetAssembledSizeVisitor getSizeVisitor;
    getSizeVisitor.setIndependentOnly(indep);
    getRoot()->execute(getSizeVisitor);
    unsigned size;
    if (id.type == sofa::core::V_COORD)
        size =  getSizeVisitor.positionSize();
    else
        size = getSizeVisitor.velocitySize();
    FullVector v(size);
    GetVectorVisitor getVec( core::MechanicalParams::defaultInstance(), &v, id);
    getVec.setIndependentOnly(indep);
    getRoot()->execute(getVec);

    Vector ve(size);
    for(size_t i=0; i<size; i++)
        ve(i)=v[i];
    return ve;
}

void setDataLink(core::objectmodel::BaseData* source, core::objectmodel::BaseData* target)
{
    target->setParent(source);
}



} // modeling



} // sofa
