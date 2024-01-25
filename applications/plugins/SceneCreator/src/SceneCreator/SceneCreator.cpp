/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SceneCreator.h"
#include <SceneCreator/config.h>

#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include "GetVectorVisitor.h"
#include "GetAssembledSizeVisitor.h"

#include <sofa/defaulttype/VecTypes.h>
using sofa::defaulttype::Vec3Types ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::DataRepository ;

#include <sofa/simulation/graph/SimpleApi.h>
using sofa::simpleapi::str ;
using sofa::simpleapi::createObject ;
using sofa::simpleapi::createChild ;

namespace sofa
{
namespace modeling {


/////////////////// IMPORTING THE DEPENDENCIES INTO THE NAMESPACE ///////////////////////////
using namespace sofa::type;
using namespace sofa::defaulttype ;

using type::vector;

using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::GetAssembledSizeVisitor ;
using sofa::simulation::GetVectorVisitor ;
using sofa::simulation::Node ;

using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::New ;

using sofa::helper::system::DataRepository ;

static sofa::simulation::Node::SPtr root = nullptr;

using sofa::core::objectmodel::BaseObject ;


Node::SPtr createRootWithCollisionPipeline(const std::string& responseType)
{
    root = simulation::getSimulation()->createNewGraph("root");
    simpleapi::createObject(root, "CollisionPipeline", {{"name","Collision Pipeline"}}) ;
    simpleapi::createObject(root, "BruteForceBroadPhase", {{"name","Broad Phase Detection"}}) ;
    simpleapi::createObject(root, "BVHNarrowPhase", {{"name","Narrow Phase Detection"}}) ;
    simpleapi::createObject(root, "MinProximityIntersection", {{"name","Proximity"},
                                                               {"alarmDistance", "0.3"},
                                                               {"contactDistance", "0.2"}}) ;

    simpleapi::createObject(root, "CollisionResponse", {
                                {"name", "Contact Manager"},
                                {"response", responseType}
                            });
    return root;
}

Node::SPtr  createEulerSolverNode(Node::SPtr parent, const std::string& name, const std::string &scheme)
{
    Node::SPtr node = simpleapi::createChild(parent, name);

    if (scheme == "Explicit")
    {
        simpleapi::createObject(parent, "RequiredPlugin", {{"name", "Sofa.Component.ODESolver.Forward"}});
        simpleapi::createObject(node, "EulerExplicitSolver", {{"name","Euler Explicit"}});
        return node ;
    }

    if (scheme == "Implicit")
    {
        simpleapi::createObject(parent, "RequiredPlugin", {{"name", "Sofa.Component.ODESolver.Backward"}});
        simpleapi::createObject(parent, "RequiredPlugin", {{"name", "Sofa.Component.LinearSolver.Iterative"}});
        simpleapi::createObject(node, "EulerImplicitSolver", {{"name","Euler Implicit"},
                                                              {"rayleighStiffness","0.01"},
                                                              {"rayleighMass", "1.0"}}) ;
        simpleapi::createObject(node, "CGLinearSolver", {{"name","Conjugate Gradient"},
                                                         {"iterations","25"},
                                                         {"threshold", "0.00001"},
                                                         {"tolerance", "0.00001"}}) ;

        return node;
    }

    if (scheme == "Implicit_SparseLDL")
    {
        simpleapi::createObject(parent, "RequiredPlugin", {{"name", "Sofa.Component.ODESolver.Backward"}});
        simpleapi::createObject(parent, "RequiredPlugin", {{"name", "Sofa.Component.LinearSolver.Direct"}});
        simpleapi::createObject(node, "EulerImplicitSolver", {{"name","Euler Implicit"},
                                                                {"rayleighStiffness","0.01"},
                                                                {"rayleighMass", "1.0"}}) ;

        simpleapi::createObject(node, "SparseLDLSolver", {{"name","Sparse LDL Solver"}});
        return node;
    }

    msg_error("SceneCreator") << scheme << " Integration Scheme not recognized.  " ;
    return node;
}


Node::SPtr createObstacle(Node::SPtr  parent, const std::string &filenameCollision,
                          const std::string filenameVisual,  const std::string& color,
                          const Deriv3& translation, const Deriv3 &rotation)
{
    Node::SPtr nodeFixed = simpleapi::createChild(parent, "Fixed") ;

    simpleapi::createObject(nodeFixed, "MeshOBJLoader", {
                                {"name","loader"},
                                {"filename", DataRepository.getFile(filenameCollision)}
                            });

    simpleapi::createObject(nodeFixed, "MeshTopology", {
                                {"name","topology"},
                                {"src", "@loader"}
                            });

    simpleapi::createObject(nodeFixed, "MechanicalObject", {
                                {"name","mecha"},
                                {"template","vec3"},
                                {"src", "@loader"},
                                {"translation", str(translation)},
                                {"rotation", str(rotation)}
                            });

    simpleapi::createObject(nodeFixed, "TriangleCollisionModel", {
                                {"name", "Collision Fixed"},
                                {"simulated", "false"},
                                {"moving", "false"},
                            });

    simpleapi::createObject(nodeFixed, "LineCollisionModel", {
                                {"name", "Collision Fixed"},
                                {"simulated", "false"},
                                {"moving", "false"},
                            });

    simpleapi::createObject(nodeFixed, "PointCollisionModel", {
                                {"name", "Collision Fixed"},
                                {"simulated", "false"},
                                {"moving", "false"},
                            });

    simpleapi::createObject(nodeFixed, "LineCollisionModel", {
                                {"name", "Collision Fixed"},
                                {"simulated", "false"},
                                {"moving", "false"},
                            });

    simpleapi::createObject(nodeFixed, "OglModel", {
                                {"name", "visual"},
                                {"filename", DataRepository.getFile(filenameVisual)},
                                {"color", color},
                                {"translation", str(translation)},
                                {"rotation", str(rotation)}
                            });
    return nodeFixed;
}


Node::SPtr createCollisionNodeVec3(Node::SPtr  parent, BaseObject::SPtr  dof,
                                   const std::string &filename,
                                   const std::vector<std::string> &elements,
                                   const Deriv3& translation, const Deriv3 &rotation)
{
    SOFA_UNUSED(dof) ;
    Node::SPtr  node = simpleapi::createChild(parent, "Collision");
    simpleapi::createObject(node, "MeshOBJLoader", {
                                {"name", "loader"},
                                {"filename", DataRepository.getFile(filename)}});

    simpleapi::createObject(node, "MeshTopology", {{"name", "loader"},
                                                   {"src", "@loader"}});

    simpleapi::createObject(node, "MechanicalObject", {{"name", "meca"},
                                                       {"src", "@loader"},
                                                       {"template","vec3"},
                                                       {"translation",str(translation)},
                                                       {"rotation",str(rotation)}});

    addCollisionModels(node, elements);

    simpleapi::createObject(node, "BarycentricMapping", {{"name", "mapping"},
                                                         {"input", "@.."},
                                                         {"output","@."}});

    return node;
}

simulation::Node::SPtr createVisualNodeVec3(simulation::Node::SPtr  parent,
                                            BaseObject::SPtr  dof,
                                            const std::string &filename, const std::string& color,
                                            const Deriv3& translation, const Deriv3 &rotation,
                                            const MappingType &mappingT)
{
    SOFA_UNUSED(dof) ;
    Node::SPtr  node = simpleapi::createChild(parent, "visualNode") ;

    std::string mappingType ;
    const std::string nameVisual="Visual";
    const std::string refVisual = "@" + nameVisual;
    const std::string refDof = "@..";

    simpleapi::createObject(node, "OglModel", {
                                {"name", nameVisual},
                                {"filename", DataRepository.getFile(filename)},
                                {"color", color},
                                {"translation", str(translation)},
                                {"rotation", str(rotation)}
                            });


    if (mappingT == MT_Barycentric) // TODO check if possible to create a baseMapping::SPtr before the switch
        mappingType = "BarycentricMapping" ;
    else if (mappingT == MT_Identity)
        mappingType = "IdentityMapping" ;
    else
    {
        msg_error("SceneCreator") << "Visual Mapping creation not possible. Mapping should be Barycentric or Identity. Found MappingType enum: " << mappingT ;
        return node ;
    }

    simpleapi::createObject(node, mappingType, {
                                {"name", nameVisual},
                                {"template", "Vec3,Vec3"},
                                {"input", refDof},
                                {"output", refVisual}});



    return node;
}



Node::SPtr createCollisionNodeRigid(Node::SPtr  parent, BaseObject::SPtr  dofRigid,
                                    const std::string &filename,
                                    const std::vector<std::string> &elements,
                                    const Deriv3& translation, const Deriv3 &rotation)
{
    const std::string refdofRigid = "@../" + dofRigid->getName();
    const std::string dofSurfName = "CollisionObject";
    const std::string refdofSurf = "@"+dofSurfName;

    Node::SPtr node=simpleapi::createChild(parent, "Collision");

    simpleapi::createObject(node, "MeshOBJLoader", {
                                {"name","loader"},
                                {"filename", DataRepository.getFile(filename)}}) ;

    simpleapi::createObject(node, "MeshTopology", {{"src", "@loader"}});

    simpleapi::createObject(node, "MechanicalObject", {
                                {"name",dofSurfName},
                                {"translation", str(translation)},
                                {"rotation", str(rotation)}});

    addCollisionModels(node, elements);

    simpleapi::createObject(node, "RigidMapping", {
                                {"input", refdofRigid},
                                {"output", refdofSurf}});

    return node;
}

Node::SPtr createVisualNodeRigid(Node::SPtr  parent, BaseObject::SPtr  dofRigid,
                                 const std::string &filename, const std::string& color,
                                 const Deriv3& translation, const Deriv3 &rotation)
{
    const std::string nameVisual="Visual";
    const std::string refVisual="@"+nameVisual;
    const std::string refdofRigid="@../"+dofRigid->getName();

    Node::SPtr node=simpleapi::createChild(parent, "Visu");

    simpleapi::createObject(node, "OglModel", {
                                {"name",nameVisual},
                                {"filename", DataRepository.getFile(filename)},
                                {"color", color},
                                {"translation", str(translation)},
                                {"rotation",str(rotation)}});

    simpleapi::createObject(node, "RigidMapping", {
                                {"name", "Mapping Visual"},
                                {"input", refdofRigid},
                                {"output", refVisual}
                            });

    return node;
}


void addCollisionModels(Node::SPtr parent, const std::vector<std::string> &elements)
{
    std::map<std::string, std::string> alias = {
        {"TriangleCollisionModel", "TriangleCollisionModel"},
        {"LineCollisionModel", "LineCollisionModel"},
        {"PointCollisionModel", "PointCollisionModel"},
        {"SphereCollisionModel", "SphereCollisionModel"},
        {"CapsuleCollisionModel", "CapsuleCollisionModel"},
        {"OBBCollisionModel", "OBBCollisionModel"},
    };

    for (auto& element : elements)
    {
        if( alias.find(element) == alias.end() )
        {
            msg_error(parent.get()) << "Unable to create collision model from '"<< element << "'" ;
            continue;
        }

        simpleapi::createObject(parent, alias[element], {{"name", element+"Collision"}}) ;
    }
}


void addTetraFEM(simulation::Node::SPtr parent, const std::string& objectName,
                 SReal totalMass, SReal young, SReal poisson)
{
    simpleapi::createObject(parent, "UniformMass", {
                                {"name",objectName + "_mass"},
                                {"totalMass", str(totalMass)},
                            });

    simpleapi::createObject(parent, "TetrahedronFEMForceField", {
                                {"name",objectName + "_FEM"},
                                {"computeGlobalMatrix", "false"},
                                {"method", "large"},
                                {"poissonRatio", str(poisson)},
                                {"youngModulus", str(young)}
                            });
}

void addTriangleFEM(simulation::Node::SPtr node, const std::string& objectName,
                    SReal totalMass, SReal young, SReal poisson)
{
    simpleapi::createObject(node, "UniformMass", {
                                {"name", objectName+"_mass"},
                                {"totalMass", str(totalMass)}});

    simpleapi::createObject(node, "TriangularFEMForceField", {
                                {"name", objectName+"_FEM"},
                                {"method", "large"},
                                {"poissonRatio", str(poisson)},
                                {"youngModulus", str(young)}
                            });
}


simulation::Node::SPtr addCube(simulation::Node::SPtr parent, const std::string& objectName,
                               const Deriv3& gridSize, SReal totalMass, SReal young, SReal poisson,
                               const Deriv3& translation, const Deriv3 &rotation, const Deriv3 &scale)
{
    //TODO(dmarchal): It is unclear to me if this message should be a msg_ (for end user)
    // or dmsg_ for developpers.
    if (parent == nullptr){
        msg_warning("SceneCreator") << "Parent node is nullptr. Returning Null Pointer." ;
        return nullptr;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Grid Size has a non positive value. Returning Null Pointer." ;
        return nullptr;
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

    const auto dofFEM = simpleapi::createObject(cube, "MechanicalObject", {
                                                    {"name", objectName+"_dof"},
                                                    {"translation", str(translation)},
                                                    {"rotation", str(rotation)},
                                                    {"scale", str(scale)}
                                                });

    // Add FEM and Mass system
    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(cube, objectName, totalMass, young, poisson);


    simpleapi::createObject(cube, "RegularGridTopology", {
                                {"name", objectName+"_grid"},
                                {"n", str(gridSize)},
                                {"min", "-0.5 -0.5 -0.5"},
                                {"max", " 0.5  0.5  0.5"}});

    // Add collisions models
    std::vector<std::string> colElements;
    colElements.push_back("TriangleCollisionModel");
    colElements.push_back("LineCollisionModel");
    colElements.push_back("PointCollisionModel");
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
    if (parent == nullptr){
        msg_warning("SceneCreator") << "Warning: parent node is nullptr. Returning Null Pointer." ;
        return nullptr;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Warning: Grid Size has a non positive value. Returning Null Pointer." ;
        return nullptr;
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

    const auto dofFEM = simpleapi::createObject(cylinder, "MechanicalObject", {
                                                    {"name", objectName+"_dof"},
                                                    {"translation", str(translation)},
                                                    {"rotation", str(rotation)},
                                                    {"scale", str(scale)}
                                                });

    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(cylinder, objectName, totalMass, young, poisson);

    simpleapi::createObject(cylinder, "CylinderGridTopology", {
                                {"name", objectName+"_grid"},
                                {"n", str(gridSize)},
                                {"radius", str(radius)},
                                {"length", str(length)},
                                {"axis", str(axis)}});


    addCollisionModels(cylinder, {"TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel"});

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
    if (parent == nullptr){
        msg_warning("SceneCreator") << "Warning: parent node is nullptr. Returning Null Pointer." ;
        return nullptr;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << "Warning: Grid Size has a non positive value. Returning Null Pointer." ;
        return nullptr;
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

    const auto dofFEM = simpleapi::createObject(sphere, "MechanicalObject", {
                                                    {"name", objectName+"_dof"},
                                                    {"translation", str(translation)},
                                                    {"rotation", str(rotation)},
                                                    {"scale", str(scale)}
                                                });

    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(sphere, objectName, totalMass, young, poisson);

    simpleapi::createObject(sphere, "SphereGridTopology", {
                                {"name", objectName+"_grid"},
                                {"n", str(gridSize)},
                                {"radius", str(radius)},
                                {"axis", str(axis)}});

    addCollisionModels(sphere, {"TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel"});
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
    if (parent == nullptr){
        msg_warning("SceneCreator") << " Parent node is nullptr. Returning Null Pointer." ;
        return nullptr;
    }

    // TODO: epernod: this should be tested in the regularGrid code to avoid crash.
    if (gridSize[0] < 1 || gridSize[1] < 1 || gridSize[2] < 1){
        msg_warning("SceneCreator") << " Grid Size has a non positive value. Returning Null Pointer." ;
        return nullptr;
    }

    // Check rigid
    bool isRigid = false;
    if (totalMass < 0.0 || young < 0.0 || poisson < 0.0)
        isRigid = true;

    // Add plane node
    sofa::simulation::Node::SPtr plane;
    if (isRigid)
        plane = parent->createChild(objectName + "_node");
    else
        plane = sofa::modeling::createEulerSolverNode(parent, objectName + "_node");

    const auto dofPlane = simpleapi::createObject(plane, "MechanicalObject", {
                                                      {"name", objectName+"_dof"},
                                                      {"translation", str(translation)},
                                                      {"rotation", str(rotation)},
                                                      {"scale", str(scale)}
                                                  });

    if (!isRigid) // Add FEM and Mass system
        addTetraFEM(plane, objectName, totalMass, young, poisson);

    simpleapi::createObject(plane, "RegularGridTopology", {
                                {"name", objectName+"_grid"},
                                {"n", str(gridSize)},
                                {"min", "-0.5 -0.5 -0.5 "},
                                {"max", " 0.5  0.5  0.5 "}
                            });

    addCollisionModels(plane, {"TriangleCollisionModel", "LineCollisionModel", "PointCollisionModel"});
    createVisualNodeVec3(plane, dofPlane, "", "green", Deriv3(), Deriv3(), MT_Identity);

    return plane;
}

simulation::Node::SPtr addRigidPlane(simulation::Node::SPtr parent, const std::string& objectName,
                                     const Deriv3& gridSize, const Deriv3& translation,
                                     const Deriv3 &rotation, const Deriv3 &scale)
{
    return addPlane(parent, objectName, gridSize, -1.f, -1.f, -1.f, translation, rotation, scale);
}

/// Create a stiff string
Node::SPtr massSpringString(Node::SPtr parent,
                            double x0, double y0, double z0, // start point,
                            double x1, double y1, double z1, // end point
                            unsigned numParticles,
                            double totalMass,
                            double stiffnessValue,
                            double dampingRatio)
{
    static unsigned numObject = 1;
    std::ostringstream oss;
    oss << "string_" << numObject++;

    Vec3d startPoint(x0,y0,z0), endPoint(x1,y1,z1);
    SReal totalLength = (endPoint-startPoint).norm();

    std::stringstream positions ;
    std::stringstream springs ;
    for( unsigned i=0; i<numParticles; i++ )
    {
        double alpha = (double)i/(numParticles-1);
        Vec3d currpos = startPoint * (1-alpha)  +  endPoint * alpha;
        positions << str(currpos) << " ";

        if(i>0)
        {
            springs << str(i-1) << " " << str(i) << " " << str(stiffnessValue)
                    << " " << str(dampingRatio) << " " << str(totalLength/(numParticles-1)) ;
        }
    }

    Node::SPtr node = simpleapi::createChild(parent, oss.str()) ;
    simpleapi::createObject(node, "MechanicalObject", {
                                {"name", oss.str()+"_DOF"},
                                {"size", str(numParticles)},
                                {"position", positions.str()}
                            });

    simpleapi::createObject(node, "UniformMass", {
                                {"name",oss.str()+"_mass"},
                                {"vertexMass", str(totalMass/numParticles)}});

    simpleapi::createObject(node, "StiffSpringForceField", {
                                {"name", oss.str()+"_spring"},
                                {"spring", springs.str()}
                            });

    return node;
}

Node::SPtr initSofa()
{
    assert(sofa::simulation::getSimulation());
    root = simulation::getSimulation()->createNewGraph("root");
    return root;
}

Node::SPtr getRoot()
{
    if(root==nullptr)
        initSofa();
    return root;
}

void initScene(Node::SPtr _root)
{
    root = _root;
    sofa::simulation::node::initRoot(_root.get());
}

Node::SPtr clearScene()
{
    if( root )
        sofa::simulation::node::unload( root );
    root = simulation::getSimulation()->createNewGraph("");
    return root;
}

void setDataLink(BaseData* source, BaseData* target)
{
    target->setParent(source);
}



} // modeling



} // sofa
