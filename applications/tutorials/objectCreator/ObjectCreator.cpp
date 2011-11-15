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

#ifndef SOFA_OBJECTCREATOR_CPP
#define SOFA_OBJECTCREATOR_CPP

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
#ifdef SOFA_DEV
#include <sofa/component/collision/BglCollisionGroupManager.h>
#endif
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/MinProximityIntersection.h>

//Including Collision Models
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/SphereModel.h>

//Including Visual Models
#include <sofa/component/visualmodel/OglModel.h>

namespace sofa
{

simulation::Node *ObjectCreator::CreateRootWithCollisionPipeline(const std::string &simulationType, const std::string& responseType)
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

#ifdef SOFA_DEV
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
    return root.get();
}

simulation::Node *ObjectCreator::CreateEulerSolverNode(simulation::Node* parent, const std::string& name, const std::string &scheme)
{
    simulation::Node* node = parent->createChild(name.c_str());

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


simulation::Node *ObjectCreator::CreateObstacle(simulation::Node* parent, const std::string &filenameCollision, const std::string filenameVisual,  const std::string& color,
        const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node* nodeFixed = parent->createChild("Fixed");

    sofa::component::loader::MeshObjLoader::SPtr loaderFixed = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loaderFixed->setFilename(sofa::helper::system::DataRepository.getFile(filenameCollision));
    nodeFixed->addObject(loaderFixed);

    component::topology::MeshTopology::SPtr meshNodeFixed = sofa::core::objectmodel::New<component::topology::MeshTopology>();
    nodeFixed->addObject(meshNodeFixed);

    MechanicalObject3::SPtr dofFixed = sofa::core::objectmodel::New<MechanicalObject3>(); dofFixed->setName("Fixed Object");
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


simulation::Node *ObjectCreator::CreateCollisionNodeVec3(simulation::Node* parent, MechanicalObject3* /*dof*/, const std::string &filename, const std::vector<std::string> &elements,
        const Deriv3& translation, const Deriv3 &rotation)
{
    //Node COLLISION
    simulation::Node* CollisionNode = parent->createChild("Collision");

    sofa::component::loader::MeshObjLoader::SPtr loader_surf = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loader_surf->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    CollisionNode->addObject(loader_surf);

    component::topology::MeshTopology::SPtr meshTorus_surf= sofa::core::objectmodel::New<component::topology::MeshTopology>();
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = sofa::core::objectmodel::New<MechanicalObject3>();  dof_surf->setName("Collision Object ");
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    AddCollisionModels(CollisionNode, elements);

    /*	BarycentricMapping3_to_3* mechaMapping = new BarycentricMapping3_to_3(dof, dof_surf);
            mechaMapping->setPathInputObject("@..");
            mechaMapping->setPathOutputObject("@.");
            CollisionNode->addObject(mechaMapping);*/

    return CollisionNode;
}

simulation::Node *ObjectCreator::CreateVisualNodeVec3(simulation::Node* parent, MechanicalObject3* dof,  const std::string &filename, const std::string& color,
        const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node* VisualNode =parent->createChild("Visu");

    const std::string nameVisual="Visual";
    const std::string refVisual = "@" + nameVisual;
    const std::string refDof = "@.." + dof->getName();
    component::visualmodel::OglModel::SPtr visual = sofa::core::objectmodel::New<component::visualmodel::OglModel>();
    visual->setName(nameVisual);
    visual->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    visual->setColor(color.c_str());
    visual->setTranslation(translation[0],translation[1],translation[2]);
    visual->setRotation(rotation[0],rotation[1],rotation[2]);
    VisualNode->addObject(visual);

    /*	BarycentricMapping3_to_Ext3* mapping = new BarycentricMapping3_to_Ext3(dof, visual);
    	mapping->setName("Mapping Visual");
    	mapping->setPathInputObject(refDof);
    	mapping->setPathOutputObject(refVisual);
            VisualNode->addObject(mapping); */

    return VisualNode;
}



simulation::Node *ObjectCreator::CreateCollisionNodeRigid(simulation::Node* parent, MechanicalObjectRigid3* dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
        const Deriv3& translation, const Deriv3 &rotation)
{
    const std::string refdofRigid = "@../" + dofRigid->getName();
    const std::string dofSurfName = "CollisionObject";
    const std::string refdofSurf = "@"+dofSurfName;
    //Node COLLISION
    simulation::Node* CollisionNode =parent->createChild("Collision");


    sofa::component::loader::MeshObjLoader::SPtr loader_surf = sofa::core::objectmodel::New<sofa::component::loader::MeshObjLoader>();
    loader_surf->setFilename(sofa::helper::system::DataRepository.getFile(filename));
    CollisionNode->addObject(loader_surf);

    component::topology::MeshTopology::SPtr meshTorus_surf= sofa::core::objectmodel::New<component::topology::MeshTopology>();
    CollisionNode->addObject(meshTorus_surf);

    MechanicalObject3::SPtr dof_surf = sofa::core::objectmodel::New<MechanicalObject3>(); dof_surf->setName(dofSurfName);
    dof_surf->setTranslation(translation[0],translation[1],translation[2]);
    dof_surf->setRotation(rotation[0],rotation[1],rotation[2]);
    CollisionNode->addObject(dof_surf);

    AddCollisionModels(CollisionNode, elements);

    /*	RigidMappingRigid3_to_3* mechaMapping = new RigidMappingRigid3_to_3(dofRigid, dof_surf);
    	mechaMapping->setPathInputObject(refdofRigid);
    	mechaMapping->setPathOutputObject(refdofSurf);
            CollisionNode->addObject(mechaMapping);*/

    return CollisionNode;
}

simulation::Node *ObjectCreator::CreateVisualNodeRigid(simulation::Node* parent, MechanicalObjectRigid3* dofRigid,  const std::string &filename, const std::string& color,
        const Deriv3& translation, const Deriv3 &rotation)
{
    simulation::Node* RigidVisualNode =parent->createChild("Visu");

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

    /*	RigidMappingRigid3_to_Ext3* mappingRigid = new RigidMappingRigid3_to_Ext3(dofRigid, visualRigid);
    	mappingRigid->setName("Mapping Visual");
    	mappingRigid->setPathInputObject(refdofRigid);
    	mappingRigid->setPathOutputObject(refVisual);
            RigidVisualNode->addObject(mappingRigid);*/
    return RigidVisualNode;
}


void ObjectCreator::AddCollisionModels(simulation::Node *CollisionNode, const std::vector<std::string> &elements)
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
    }
}

}

#endif
