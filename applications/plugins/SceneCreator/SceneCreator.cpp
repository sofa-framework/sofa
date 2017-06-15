/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#define SOFA_SIMPLEOBJECTCREATOR_CPP

#include "SceneCreator.h"
#include <SofaGeneral/config.h>

#include <sofa/helper/system/SetDirectory.h>


//Including Simulation
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include "GetVectorVisitor.h"
#include "GetAssembledSizeVisitor.h"

//Including Solvers and linear algebra
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

#ifdef SOFA_HAVE_METIS
#include <SofaSparseSolver/SparseLDLSolver.h>
#endif

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

#ifndef SOFA_NO_OPENGL
#include <SofaOpenglVisual/OglModel.h>
#else
#include <SofaBaseVisual/VisualModelImpl.h>
#endif

#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseMechanics/IdentityMapping.h>
#include <SofaBaseMechanics/BarycentricMapping.h>

#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaSimpleFem/TetrahedronFEMForceField.h>
#include <SofaMiscFem/TriangularFEMForceField.h>

#include <SofaGeneralTopology/CylinderGridTopology.h>
#include <SofaGeneralTopology/SphereGridTopology.h>

#include <SofaBaseLinearSolver/FullVector.h>

namespace sofa
{
namespace modeling {
/////////////////// IMPORTING THE DEPENDENCIES INTO THE NAMESPACE ///////////////////////////
using namespace sofa::defaulttype ;
using namespace sofa::component::collision ;

using helper::vector;
using sofa::component::linearsolver::GraphScatteredMatrix ;
using sofa::component::linearsolver::GraphScatteredVector ;
using sofa::component::linearsolver::CGLinearSolver ;

using sofa::component::odesolver::EulerImplicitSolver ;
using sofa::component::odesolver::EulerSolver ;

using sofa::component::loader::MeshObjLoader ;
using sofa::component::topology::MeshTopology ;

using sofa::component::topology::RegularGridTopology ;
using sofa::component::topology::CylinderGridTopology ;
using sofa::component::topology::SphereGridTopology;


/// Dense state vector deriving from BaseVector, used to access data in the scene graph
typedef component::linearsolver::FullVector<SReal> FullVector;



#ifndef SOFA_NO_OPENGL
using sofa::component::visualmodel::OglModel ;
#else
using sofa::component::visualmodel::VisualModelImpl;
#endif

using sofa::component::mapping::BarycentricMapping ;
using sofa::component::mapping::RigidMapping ;
using sofa::component::mapping::IdentityMapping ;

using sofa::component::interactionforcefield::StiffSpringForceField ;
using sofa::component::forcefield::TetrahedronFEMForceField;
using sofa::component::forcefield::TriangularFEMForceField;
using sofa::component::mass::UniformMass ;

using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::GetAssembledSizeVisitor ;
using sofa::simulation::GetVectorVisitor ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;

using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::New ;

using sofa::helper::system::DataRepository ;


/////////////////// ALIAS FOR THE DIFFERENT TEMPLATE WE NEED TO USE ///////////////////////////
typedef CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >    CGLinearSolverGraph;
typedef UniformMass<Vec3Types, SReal>                                   UniformMass3;
typedef StiffSpringForceField<Vec3Types >                               StiffSpringForceField3;
typedef TetrahedronFEMForceField<Vec3Types>                             TetrahedronFEMForceField3;
typedef TriangularFEMForceField<Vec3Types>                              TriangularFEMForceField3;

typedef IdentityMapping<Vec3Types, ExtVec3fTypes>       IdentityMapping3_to_Ext3;
typedef BarycentricMapping<Vec3Types, Vec3Types >       BarycentricMapping3_to_3;
typedef BarycentricMapping<Vec3Types, ExtVec3fTypes>    BarycentricMapping3_to_Ext3;
typedef RigidMapping<Rigid3Types, Vec3Types >           RigidMappingRigid3_to_3;
typedef RigidMapping<Rigid3Types, ExtVec3fTypes >       RigidMappingRigid3_to_Ext3;


#ifndef SOFA_NO_OPENGL
typedef component::visualmodel::OglModel VisualModelType;
#else
typedef component::visualmodel::VisualModelImpl VisualModelType;
#endif

static sofa::simulation::Node::SPtr root = NULL;

Node::SPtr createRootWithCollisionPipeline(const std::string& responseType)
{
    root = simulation::getSimulation()->createNewGraph("root");

    //Components for collision management
    //------------------------------------
    //--> adding collision pipeline
    DefaultPipeline::SPtr collisionPipeline = New<DefaultPipeline>();
    collisionPipeline->setName("Collision Pipeline");
    root->addObject(collisionPipeline);

    //--> adding collision detection system
    BruteForceDetection::SPtr detection = New<BruteForceDetection>();
    detection->setName("Detection");
    root->addObject(detection);

    //--> adding component to detection intersection of elements
    MinProximityIntersection::SPtr detectionProximity = New<MinProximityIntersection>();
    detectionProximity->setName("Proximity");
    detectionProximity->setAlarmDistance(0.3);   //warning distance
    detectionProximity->setContactDistance(0.2); //min distance before setting a spring to create a repulsion
    root->addObject(detectionProximity);

    //--> adding contact manager
    DefaultContactManager::SPtr contactManager = New<DefaultContactManager>();
    contactManager->setName("Contact Manager");
    contactManager->setDefaultResponseType(responseType);
    root->addObject(contactManager);

    //--> adding component to handle groups of collision.
    DefaultCollisionGroupManager::SPtr collisionGroupManager = New<DefaultCollisionGroupManager>();
    collisionGroupManager->setName("Collision Group Manager");
    root->addObject(collisionGroupManager);

    return root;
}

Node::SPtr  createEulerSolverNode(Node::SPtr parent, const std::string& name, const std::string &scheme)
{
    Node::SPtr  node = parent->createChild(name.c_str());

    if (scheme == "Implicit")
    {
        EulerImplicitSolver::SPtr solver = New<EulerImplicitSolver>();
        CGLinearSolverGraph::SPtr linear = New<CGLinearSolverGraph>();
        solver->setName("Euler Implicit");
        solver->f_rayleighStiffness.setValue(0.01);
        solver->f_rayleighMass.setValue(1);

        linear->setName("Conjugate Gradient");
        linear->f_maxIter.setValue(25); //iteration maxi for the CG
        linear->f_smallDenominatorThreshold.setValue(1e-05);
        linear->f_tolerance.setValue(1e-05);

        node->addObject(solver);
        node->addObject(linear);
    }
    else if (scheme == "Explicit")
    {
        EulerSolver::SPtr solver = New<EulerSolver>();
        solver->setName("Euler Explicit");
        node->addObject(solver);
    }

    else if (scheme == "Implicit_SparseLDL")
    {
#ifdef SOFA_HAVE_METIS
        typedef Mat<3,3,double> Block33_double;

        using sofa::component::linearsolver::FullVector;
        using sofa::component::linearsolver::SparseLDLSolver;
        using sofa::component::linearsolver::CompressedRowSparseMatrix;

        typedef FullVector<double> FullVectorDouble;
        typedef CompressedRowSparseMatrix< Block33_double > CompressedRowSparseMatrix_33;
        typedef SparseLDLSolver< CompressedRowSparseMatrix_33 , FullVectorDouble > SparseLDLSolver_33;

        EulerImplicitSolver::SPtr solver = New<EulerImplicitSolver>();
        SparseLDLSolver_33::SPtr linear = New<SparseLDLSolver_33>();
        solver->setName("Euler Implicit");
        solver->f_rayleighStiffness.setValue(0.01);
        solver->f_rayleighMass.setValue(1);

        linear->setName("Sparse LDL Solver");
        node->addObject(solver);
        node->addObject(linear);
#else
        msg_error("SceneCreator") << "Unable to create a scene because this verson of sofa has not been compiled with SparseLDLSolver. " ;
#endif
    }

    else
    {
        msg_error("SceneCreator") << scheme << " Integration Scheme not recognized.  " ;
    }
    return node;
}


Node::SPtr createObstacle(Node::SPtr  parent, const std::string &filenameCollision, const std::string filenameVisual,  const std::string& color,
                                                           const Deriv3& translation, const Deriv3 &rotation)
{
    Node::SPtr  nodeFixed = parent->createChild("Fixed");

    MeshObjLoader::SPtr loaderFixed = New<MeshObjLoader>();
    loaderFixed->setName("loader");
    loaderFixed->setFilename(DataRepository.getFile(filenameCollision));
    loaderFixed->load();
    nodeFixed->addObject(loaderFixed);

    MeshTopology::SPtr meshNodeFixed = New<MeshTopology>();
    meshNodeFixed->setSrc("@"+loaderFixed->getName(), loaderFixed.get());
    nodeFixed->addObject(meshNodeFixed);

    MechanicalObject3::SPtr dofFixed = New<MechanicalObject3>(); dofFixed->setName("Fixed Object");
    dofFixed->setSrc("@"+loaderFixed->getName(), loaderFixed.get());
    dofFixed->setTranslation(translation[0],translation[1],translation[2]);
    dofFixed->setRotation(rotation[0],rotation[1],rotation[2]);
    nodeFixed->addObject(dofFixed);

    TriangleModel::SPtr triangleFixed = New<TriangleModel>(); triangleFixed->setName("Collision Fixed");
    triangleFixed->setSimulated(false); //Not simulated, fixed object
    triangleFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(triangleFixed);
    LineModel::SPtr LineFixed = New<LineModel>(); LineFixed->setName("Collision Fixed");
    LineFixed->setSimulated(false); //Not simulated, fixed object
    LineFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(LineFixed);
    PointModel::SPtr PointFixed = New<PointModel>(); PointFixed->setName("Collision Fixed");
    PointFixed->setSimulated(false); //Not simulated, fixed object
    PointFixed->setMoving(false);    //No extern events
    nodeFixed->addObject(PointFixed);

    VisualModelType::SPtr visualFixed = sofa::core::objectmodel::New<VisualModelType>();
    visualFixed->setName("visual");
    visualFixed->setFilename(DataRepository.getFile(filenameVisual));
    visualFixed->setColor(color);
    visualFixed->setTranslation(translation[0],translation[1],translation[2]);
    visualFixed->setRotation(rotation[0],rotation[1],rotation[2]);
    nodeFixed->addObject(visualFixed);
    return nodeFixed;
}


Node::SPtr createCollisionNodeVec3(Node::SPtr  parent, MechanicalObject3::SPtr  dof, const std::string &filename, const std::vector<std::string> &elements,
                                                                    const Deriv3& translation, const Deriv3 &rotation)
{
    //Node COLLISION
    Node::SPtr  CollisionNode = parent->createChild("Collision");

    MeshObjLoader::SPtr loader_surf = New<MeshObjLoader>();
    loader_surf->setName("loader");
    loader_surf->setFilename(DataRepository.getFile(filename));
    loader_surf->load();
    CollisionNode->addObject(loader_surf);

    MeshTopology::SPtr meshTorus_surf= New<MeshTopology>();
    meshTorus_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = New<MechanicalObject3>();  dof_surf->setName("Collision Object ");
    dof_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    addCollisionModels(CollisionNode, elements);

    BarycentricMapping3_to_3::SPtr mechaMapping = New<BarycentricMapping3_to_3>();
    mechaMapping->setModels(dof.get(), dof_surf.get());
    mechaMapping->setPathInputObject("@..");
    mechaMapping->setPathOutputObject("@.");
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

simulation::Node::SPtr createVisualNodeVec3(simulation::Node::SPtr  parent, MechanicalObject3::SPtr  dof,  const std::string &filename, const std::string& color,
                                                                 const Deriv3& translation, const Deriv3 &rotation, const MappingType &mappingT)
{
    Node::SPtr  VisualNode =parent->createChild("Visu");

    const std::string nameVisual="Visual";
    const std::string refVisual = "@" + nameVisual;
    const std::string refDof = "@..";// + dof->getName();
    VisualModelType::SPtr visual = sofa::core::objectmodel::New<VisualModelType>();
    visual->setName(nameVisual);
    visual->setFilename(DataRepository.getFile(filename));
    visual->setColor(color.c_str());
    visual->setTranslation(translation[0],translation[1],translation[2]);
    visual->setRotation(rotation[0],rotation[1],rotation[2]);
    VisualNode->addObject(visual);

    if (mappingT == MT_Barycentric) // TODO check if possible to create a baseMapping::SPtr before the switch
    {
        BarycentricMapping3_to_Ext3::SPtr mapping = sofa::core::objectmodel::New<BarycentricMapping3_to_Ext3>();
        mapping->setModels(dof.get(), visual.get());
        mapping->setName("Mapping Visual");
        mapping->setPathInputObject(refDof);
        mapping->setPathOutputObject(refVisual);
        VisualNode->addObject(mapping);
    }
    else if (mappingT == MT_Identity)
    {
        IdentityMapping3_to_Ext3::SPtr mapping = sofa::core::objectmodel::New<IdentityMapping3_to_Ext3>();
        mapping->setModels(dof.get(), visual.get());
        mapping->setName("Mapping Visual");
        mapping->setPathInputObject(refDof);
        mapping->setPathOutputObject(refVisual);
        VisualNode->addObject(mapping);
    }
    else
        msg_error("SceneCreator") << "Visual Mapping creation not possible. Mapping should be Barycentric or Identity. Found MappingType enum: " << mappingT ;

    return VisualNode;
}



Node::SPtr createCollisionNodeRigid(Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
                                                                     const Deriv3& translation, const Deriv3 &rotation)
{
    const std::string refdofRigid = "@../" + dofRigid->getName();
    const std::string dofSurfName = "CollisionObject";
    const std::string refdofSurf = "@"+dofSurfName;
    //Node COLLISION
    Node::SPtr  CollisionNode =parent->createChild("Collision");


    MeshObjLoader::SPtr loader_surf = New<MeshObjLoader>();
    loader_surf->setName("loader");
    loader_surf->setFilename(DataRepository.getFile(filename));
    loader_surf->load();
    CollisionNode->addObject(loader_surf);

    MeshTopology::SPtr meshTorus_surf= New<MeshTopology>();
    meshTorus_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = New<MechanicalObject3>(); dof_surf->setName(dofSurfName);
    //    dof_surf->setSrc("@"+loader_surf->getName(), loader_surf.get());
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    addCollisionModels(CollisionNode, elements);

    RigidMappingRigid3_to_3::SPtr mechaMapping = New<RigidMappingRigid3_to_3>();
    mechaMapping->setModels(dofRigid.get(), dof_surf.get());
    mechaMapping->setPathInputObject(refdofRigid);
    mechaMapping->setPathOutputObject(refdofSurf);
    CollisionNode->addObject(mechaMapping);

    return CollisionNode;
}

Node::SPtr createVisualNodeRigid(Node::SPtr  parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::string& color,
                                                                  const Deriv3& translation, const Deriv3 &rotation)
{
    Node::SPtr  RigidVisualNode =parent->createChild("Visu");

    const std::string nameVisual="Visual";
    const std::string refVisual="@"+nameVisual;
    const std::string refdofRigid="@../"+dofRigid->getName();
    VisualModelType::SPtr visualRigid = sofa::core::objectmodel::New<VisualModelType>();
    visualRigid->setName(nameVisual);
    visualRigid->setFilename(DataRepository.getFile(filename));
    visualRigid->setColor(color);
    visualRigid->setTranslation(translation[0],translation[1],translation[2]);
    visualRigid->setRotation(rotation[0],rotation[1],rotation[2]);
    RigidVisualNode->addObject(visualRigid);

    RigidMappingRigid3_to_Ext3::SPtr mappingRigid = New<RigidMappingRigid3_to_Ext3>();
    mappingRigid->setModels(dofRigid.get(), visualRigid.get());
    mappingRigid->setName("Mapping Visual");
    mappingRigid->setPathInputObject(refdofRigid);
    mappingRigid->setPathOutputObject(refVisual);
    RigidVisualNode->addObject(mappingRigid);
    return RigidVisualNode;
}


void addCollisionModels(Node::SPtr CollisionNode, const std::vector<std::string> &elements)
{
    for (unsigned int i=0; i<elements.size(); ++i)
    {
        if (elements[i] == "Triangle")
        {
            TriangleModel::SPtr triangle = New<TriangleModel>();  triangle->setName("TriangleCollision");
            CollisionNode->addObject(triangle);
        }
        else if(elements[i] == "Line")
        {
            LineModel::SPtr line = New<LineModel>();  line->setName("LineCollision");
            CollisionNode->addObject(line);
        }
        else if (elements[i] == "Point")
        {
            PointModel::SPtr point = New<PointModel>();  point->setName("PointCollision");
            CollisionNode->addObject(point);
        }
        else if (elements[i] == "Sphere")
        {
            SphereModel::SPtr point = New<SphereModel>();  point->setName("SphereCollision");
            CollisionNode->addObject(point);
        }
        else if(elements[i] == "Capsule"){
            CapsuleModel::SPtr capsule = New<CapsuleModel>();  capsule->setName("CapsuleCollision");
            CollisionNode->addObject(capsule);
        }
        else if(elements[i] == "OBB"){
            OBBModel::SPtr obb = New<OBBModel>();  obb->setName("OBBCollision");
            CollisionNode->addObject(obb);
        }
    }
}


void addTetraFEM(simulation::Node::SPtr currentNode, const std::string& objectName,
                 SReal totalMass, SReal young, SReal poisson)
{
    // Add Mass
    UniformMass3::SPtr uniMassSpring = sofa::core::objectmodel::New<UniformMass3>();
    uniMassSpring->setTotalMass(totalMass);
    uniMassSpring->setName(objectName + "_mass");
    currentNode->addObject(uniMassSpring);

    // Add FEM
    TetrahedronFEMForceField3::SPtr tetraFEMFF = sofa::core::objectmodel::New<TetrahedronFEMForceField3>();
    tetraFEMFF->setName(objectName + "_FEM");
    tetraFEMFF->setComputeGlobalMatrix(false);
    tetraFEMFF->setMethod("large");
    tetraFEMFF->setPoissonRatio(poisson);
    tetraFEMFF->setYoungModulus(young);
    currentNode->addObject(tetraFEMFF);
}

void addTriangleFEM(simulation::Node::SPtr currentNode, const std::string& objectName,
                    SReal totalMass, SReal young, SReal poisson)
{
    // Add Mass
    UniformMass3::SPtr uniMassSpring = sofa::core::objectmodel::New<UniformMass3>();
    uniMassSpring->setTotalMass(totalMass);
    uniMassSpring->setName(objectName + "_mass");
    currentNode->addObject(uniMassSpring);

    // Add FEM
    TriangularFEMForceField3::SPtr triFEMFF = sofa::core::objectmodel::New<TriangularFEMForceField3>();
    triFEMFF->setName(objectName + "_FEM");
    triFEMFF->setMethod("large");
    triFEMFF->setPoisson(poisson);
    triFEMFF->setYoung(young);
    currentNode->addObject(triFEMFF);
}


simulation::Node::SPtr addCube(simulation::Node::SPtr parent, const std::string& objectName,
                               const Deriv3& gridSize, SReal totalMass, SReal young, SReal poisson,
                               const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    //TODO(dmarchal): It is unclear to me if this message should be a msg_ (for end user)
    // or dmsg_ for developpers.
    if (parent == NULL){
        msg_warning("SceneCreator") << "Parent node is NULL. Returning Null Pointer." ;
        return NULL;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Grid Size has a non positive value. Returning Null Pointer." ;
        return NULL;
    }

    // Check rigid
    bool isRigid = false;
    if (totalMass < 0.0 || young < 0.0 || poisson < 0.0)
        isRigid = true;

    // Add Cube Node
    sofa::simulation::Node::SPtr cube;
    if (isRigid)
        cube = parent->createChild(objectName + "_node");
    else
        cube = sofa::modeling::createEulerSolverNode(parent, objectName + "_node");


    // Add mecaObj
    MechanicalObject3::SPtr dofFEM = sofa::core::objectmodel::New<MechanicalObject3>(); dofFEM->setName(objectName + "_dof");
    dofFEM->setTranslation(translation[0],translation[1],translation[2]);
    dofFEM->setRotation(rotation[0],rotation[1],rotation[2]);
    dofFEM->setScale(scale[0],scale[1],scale[2]);
    cube->addObject(dofFEM);

    // Add FEM and Mass system
    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(cube, objectName, totalMass, young, poisson);

    // Add topo
    RegularGridTopology::SPtr grid = sofa::core::objectmodel::New<RegularGridTopology>();
    grid->setName(objectName + "_grid");
    grid->setSize(gridSize[0], gridSize[1], gridSize[2]);
    grid->setPos(-0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f); // by default object is centered and volume equal to 1 unit, use dof modifier to change the scale/position/rotation
    cube->addObject(grid);


    // Add collisions models
    std::vector<std::string> colElements;
    colElements.push_back("Triangle");
    colElements.push_back("Line");
    colElements.push_back("Point");
    sofa::modeling::addCollisionModels(cube, colElements);

    //Node VISUAL
    createVisualNodeVec3(cube, dofFEM, "", "red", Deriv3(), Deriv3(), MT_Identity);

    return cube;
}


simulation::Node::SPtr addRigidCube(simulation::Node::SPtr parent, const std::string& objectName,
                                    const Deriv3& gridSize,
                                    const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    return addCube(parent, objectName, gridSize, -1.f, -1.f, -1.f, translation, rotation, scale);
}



simulation::Node::SPtr addCylinder(simulation::Node::SPtr parent, const std::string& objectName,
                                   const Deriv3& gridSize, const Deriv3& axis, SReal radius, SReal length,
                                   SReal totalMass, SReal young, SReal poisson,
                                   const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    //TODO(dmarchal): It is unclear to me if this message should be a msg_ (for end user)
    // or dmsg_ for developpers.
    if (parent == NULL){
        msg_warning("SceneCreator") << "Warning: parent node is NULL. Returning Null Pointer." ;
        return NULL;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Warning: Grid Size has a non positive value. Returning Null Pointer." ;
        return NULL;
    }

    // Check rigid
    bool isRigid = false;
    if (totalMass < 0.0 || young < 0.0 || poisson < 0.0)
        isRigid = true;

    // Add Cylinder Node
    sofa::simulation::Node::SPtr cylinder;
    if (isRigid)
        cylinder = parent->createChild(objectName + "_node");
    else
        cylinder = sofa::modeling::createEulerSolverNode(parent, objectName + "_node");


    // Add mecaObj
    MechanicalObject3::SPtr dofFEM = sofa::core::objectmodel::New<MechanicalObject3>(); dofFEM->setName(objectName + "_dof");
    dofFEM->setTranslation(translation[0],translation[1],translation[2]);
    dofFEM->setRotation(rotation[0],rotation[1],rotation[2]);
    dofFEM->setScale(scale[0],scale[1],scale[2]);
    cylinder->addObject(dofFEM);

    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(cylinder, objectName, totalMass, young, poisson);

    // Add topo
    CylinderGridTopology::SPtr grid = sofa::core::objectmodel::New<CylinderGridTopology>();
    grid->setName(objectName + "_grid");
    grid->setSize(gridSize[0], gridSize[1], gridSize[2]);
    grid->setRadius(radius);
    grid->setLength(length);
    grid->setAxis(axis[0], axis[1], axis[2]);
    cylinder->addObject(grid);


    // Add collisions models
    std::vector<std::string> colElements;
    colElements.push_back("Triangle");
    colElements.push_back("Line");
    colElements.push_back("Point");
    sofa::modeling::addCollisionModels(cylinder, colElements);

    //Node VISUAL
    createVisualNodeVec3(cylinder, dofFEM, "", "red", Deriv3(), Deriv3(), MT_Identity);

    return cylinder;
}


simulation::Node::SPtr addRigidCylinder(simulation::Node::SPtr parent, const std::string& objectName,
                                   const Deriv3& gridSize, const Deriv3& axis, SReal radius, SReal length,
                                   const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    return addCylinder(parent, objectName, gridSize, axis, radius, length, -1.f, -1.f, -1.f, translation, rotation, scale);
}

simulation::Node::SPtr addSphere(simulation::Node::SPtr parent, const std::string& objectName,
                                   const Deriv3& gridSize, const Deriv3& axis, SReal radius,
                                   SReal totalMass, SReal young, SReal poisson,
                                   const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    //TODO(dmarchal): It is unclear to me if this message should be a msg_ (for end user)
    // or dmsg_ for developpers.
    if (parent == NULL){
        msg_warning("SceneCreator") << "Warning: parent node is NULL. Returning Null Pointer." ;
        return NULL;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Warning: Grid Size has a non positive value. Returning Null Pointer." ;
        return NULL;
    }

    // Check rigid
    bool isRigid = false;
    if (totalMass < 0.0 || young < 0.0 || poisson < 0.0)
        isRigid = true;

    // Add Sphere Node
    sofa::simulation::Node::SPtr sphere;
    if (isRigid)
        sphere = parent->createChild(objectName + "_node");
    else
        sphere = sofa::modeling::createEulerSolverNode(parent, objectName + "_node");


    // Add mecaObj
    MechanicalObject3::SPtr dofFEM = sofa::core::objectmodel::New<MechanicalObject3>(); dofFEM->setName(objectName + "_dof");
    dofFEM->setTranslation(translation[0],translation[1],translation[2]);
    dofFEM->setRotation(rotation[0],rotation[1],rotation[2]);
    dofFEM->setScale(scale[0],scale[1],scale[2]);
    sphere->addObject(dofFEM);

    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(sphere, objectName, totalMass, young, poisson);

    // Add topo
    SphereGridTopology::SPtr grid = sofa::core::objectmodel::New<SphereGridTopology>();
    grid->setName(objectName + "_grid");
    grid->setSize(gridSize[0], gridSize[1], gridSize[2]);
    grid->setRadius(radius);
    grid->setAxis(axis[0], axis[1], axis[2]);
    sphere->addObject(grid);


    // Add collisions models
    std::vector<std::string> colElements;
    colElements.push_back("Triangle");
    colElements.push_back("Line");
    colElements.push_back("Point");
    sofa::modeling::addCollisionModels(sphere, colElements);

    //Node VISUAL
    createVisualNodeVec3(sphere, dofFEM, "", "red", Deriv3(), Deriv3(), MT_Identity);

    return sphere;
}


simulation::Node::SPtr addRigidSphere(simulation::Node::SPtr parent, const std::string& objectName,
                                      const Deriv3& gridSize, const Deriv3& axis, SReal radius,
                                      const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    return addSphere(parent, objectName, gridSize, axis, radius, -1.f, -1.f, -1.f, translation, rotation, scale);
}


simulation::Node::SPtr addPlane(simulation::Node::SPtr parent, const std::string& objectName,
                                const Deriv3& gridSize, SReal totalMass, SReal young, SReal poisson,
                                const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    //TODO(dmarchal): It is unclear to me if this message should be a msg_ (for end user)
    // or dmsg_ for developpers.
    if (parent == NULL){
        msg_warning("SceneCreator") << " Parent node is NULL. Returning Null Pointer." ;
        return NULL;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << " Grid Size has a non positive value. Returning Null Pointer." ;
        return NULL;
    }

    // Check rigid
    bool isRigid = false;
    if (totalMass < 0.0 || young < 0.0 || poisson < 0.0)
        isRigid = true;

    // Add plane node
    sofa::simulation::Node::SPtr planeNode;
    if (isRigid)
        planeNode = parent->createChild(objectName + "_node");
    else
        planeNode = sofa::modeling::createEulerSolverNode(parent, objectName + "_node");

    // Add mecaObj
    MechanicalObject3::SPtr dofPlane = sofa::core::objectmodel::New<MechanicalObject3>(); dofPlane->setName("FEM Object");
    dofPlane->setTranslation(translation[0],translation[1],translation[2]);
    dofPlane->setRotation(rotation[0],rotation[1],rotation[2]);
    dofPlane->setScale(scale[0],scale[1],scale[2]);
    planeNode->addObject(dofPlane);

    if (!isRigid) // Add FEM and Mass system
        addTriangleFEM(planeNode, objectName, totalMass, young, poisson);

    // Add topo
    sofa::component::topology::RegularGridTopology::SPtr gridPlane = sofa::core::objectmodel::New<sofa::component::topology::RegularGridTopology>();
    gridPlane->setSize(gridSize[0], gridSize[1], gridSize[2]);
    gridPlane->setPos(-0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f);
    planeNode->addObject(gridPlane);

    std::vector<std::string> colElements;
    colElements.push_back("Triangle");
    colElements.push_back("Line");
    colElements.push_back("Point");
    sofa::modeling::addCollisionModels(planeNode, colElements);

    //Node VISUAL
    createVisualNodeVec3(planeNode, dofPlane, "", "green", Deriv3(), Deriv3(), MT_Identity);

    return planeNode;
}

simulation::Node::SPtr addRigidPlane(simulation::Node::SPtr parent, const std::string& objectName, const Deriv3& gridSize,
                                     const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    return addPlane(parent, objectName, gridSize, -1.f, -1.f, -1.f, translation, rotation, scale);
}

/// Create a stiff string
Node::SPtr massSpringString
(
        Node::SPtr parent,
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
    mass->d_mass.setValue( totalMass/numParticles );

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
    root = simulation::getSimulation()->createNewGraph("root");
    return root;
}


void initScene(Node::SPtr _root)
{
    root = _root;
    sofa::simulation::getSimulation()->init(root.get());
}

Node::SPtr clearScene()
{
    if( root )
        Simulation::theSimulation->unload( root );
    root = Simulation::theSimulation->createNewGraph("");
    return root;
}


Vector getVector( core::ConstVecId id, bool indep )
{
    GetAssembledSizeVisitor getSizeVisitor;
    getSizeVisitor.setIndependentOnly(indep);
    root->execute(getSizeVisitor);
    unsigned size;
    if (id.type == sofa::core::V_COORD)
        size =  getSizeVisitor.positionSize();
    else
        size = getSizeVisitor.velocitySize();
    FullVector v(size);
    GetVectorVisitor getVec( core::MechanicalParams::defaultInstance(), &v, id);
    getVec.setIndependentOnly(indep);
    root->execute(getVec);

    Vector ve(size);
    for(size_t i=0; i<size; i++)
        ve(i)=v[i];
    return ve;
}

void setDataLink(BaseData* source, BaseData* target)
{
    target->setParent(source);
}



} // modeling



} // sofa
