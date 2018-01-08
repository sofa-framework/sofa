/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

#ifndef SOFA_SCENECREATOR_H
#define SOFA_SCENECREATOR_H

#include <SceneCreator/config.h>
#include <string>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/defaulttype/Vec3Types.h>

/// @warning this can only manage one scene at a time
/// (root singleton)
namespace sofa
{
namespace modeling
{
using sofa::core::objectmodel::BaseObject ;

typedef SReal Scalar;
typedef sofa::defaulttype::Vec3Types::Deriv Deriv3;
typedef sofa::defaulttype::Vec<3,SReal> Vec3;
typedef sofa::defaulttype::Vec<1,SReal> Vec1;

typedef enum
{
    MT_Barycentric = 0,
    MT_Rigid,
    MT_Identity
} MappingType;

SOFA_SCENECREATOR_API simulation::NodeSPtr createRootWithCollisionPipeline(const std::string &responseType=std::string("default"));

SOFA_SCENECREATOR_API simulation::NodeSPtr createEulerSolverNode(simulation::NodeSPtr parent,
                                                                   const std::string& name,
                                                                   const std::string &integrationScheme=std::string("Implicit"));


SOFA_SCENECREATOR_API simulation::NodeSPtr createObstacle(simulation::NodeSPtr parent,
                                                            const std::string &filenameCollision,
                                                            const std::string filenameVisual,
                                                            const std::string& color,
                                                            const Deriv3& translation=Deriv3(),
                                                            const Deriv3 &rotation=Deriv3());

/// Create a collision node using Barycentric Mapping, using a 3d model specified by filename.
/// elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
/// an initial transformation can be performed
SOFA_SCENECREATOR_API simulation::NodeSPtr createCollisionNodeVec3(simulation::NodeSPtr parent, BaseObject::SPtr dof,
                                                                     const std::string &filename,
                                                                     const std::vector<std::string> &elements,
                                                                     const Deriv3& translation=Deriv3(),
                                                                     const Deriv3 &rotation=Deriv3());

SOFA_SCENECREATOR_API simulation::NodeSPtr createVisualNodeVec3(simulation::NodeSPtr parent, BaseObject::SPtr dof,
                                                                  const std::string &filename, const std::string& color,
                                                                  const Deriv3& translation=Deriv3(),
                                                                  const Deriv3 &rotation=Deriv3(),
                                                                  const MappingType& mappingT=MT_Barycentric);


/// Create a collision node using Rigid Mapping, using a 3d model specified by filename.
/// elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
/// an initial transformation can be performed
SOFA_SCENECREATOR_API simulation::NodeSPtr createCollisionNodeRigid(simulation::NodeSPtr parent,
                                                                      BaseObject::SPtr dofRigid,
                                                                      const std::string &filename,
                                                                      const std::vector<std::string> &elements,
                                                                      const Deriv3& translation=Deriv3(),
                                                                      const Deriv3 &rotation=Deriv3());

SOFA_SCENECREATOR_API simulation::NodeSPtr createVisualNodeRigid(simulation::NodeSPtr parent,
                                                                   BaseObject::SPtr  dofRigid,
                                                                   const std::string &filename,
                                                                   const std::string& color,
                                                                   const Deriv3& translation=Deriv3(),
                                                                   const Deriv3 &rotation=Deriv3());

SOFA_SCENECREATOR_API simulation::NodeSPtr createGridScene(Vec3 startPoint, Vec3 endPoint,
                                                             unsigned numX, unsigned numY, unsigned numZ,
                                                             double totalMass, double stiffnessValue=1.0,
                                                             double dampingRatio=0 );


SOFA_SCENECREATOR_API void addCollisionModels(simulation::NodeSPtr CollisionNode,
                                              const std::vector<std::string> &elements);


/// Create 3D objects, using mechanical Obj, grid topology and visualisation inside one node
/// By default object is centered and volume equal to 1 unit, use dof modifier to change the scale/position/rotation
SOFA_SCENECREATOR_API simulation::NodeSPtr addCube(simulation::NodeSPtr parent, const std::string& objectName,
                                                     const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                     SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3,
                                                     const Deriv3& translation=Deriv3(),
                                                     const Deriv3 &rotation=Deriv3(),
                                                     const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addRigidCube(simulation::NodeSPtr parent, const std::string& objectName,
                                                          const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                          const Deriv3& translation=Deriv3(),
                                                          const Deriv3 &rotation=Deriv3(),
                                                          const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addCylinder(simulation::NodeSPtr parent, const std::string& objectName,
                                                         const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                         const Deriv3& axis = Deriv3(0, 1, 0), SReal radius = 1.0, SReal length = 1.0,
                                                         SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3,
                                                         const Deriv3& translation=Deriv3(),
                                                         const Deriv3 &rotation=Deriv3(),
                                                         const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addRigidCylinder(simulation::NodeSPtr parent,
                                                              const std::string& objectName,
                                                              const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                              const Deriv3& axis = Deriv3(0, 1, 0), SReal radius = 1.0, SReal length = 1.0,
                                                              const Deriv3& translation=Deriv3(),
                                                              const Deriv3 &rotation=Deriv3(),
                                                              const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addSphere(simulation::NodeSPtr parent, const std::string& objectName,
                                                         const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                         const Deriv3& axis = Deriv3(0, 1, 0), SReal radius = 1.0,
                                                         SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3,
                                                         const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3(), const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addRigidSphere(simulation::NodeSPtr parent, const std::string& objectName,
                                                              const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                              const Deriv3& axis = Deriv3(0, 1, 0), SReal radius = 1.0,
                                                              const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3(), const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));


SOFA_SCENECREATOR_API simulation::NodeSPtr addPlane(simulation::NodeSPtr parent,
                                                      const std::string& objectName,
                                                      const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                      SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3,
                                                      const Deriv3& translation=Deriv3(),
                                                      const Deriv3 &rotation=Deriv3(),
                                                      const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));

SOFA_SCENECREATOR_API simulation::NodeSPtr addRigidPlane(simulation::NodeSPtr parent,
                                                           const std::string& objectName,
                                                           const Deriv3& gridSize=Deriv3(10, 10, 10),
                                                           const Deriv3& translation=Deriv3(),
                                                           const Deriv3 &rotation=Deriv3(),
                                                           const Deriv3 &scale=Deriv3(1.0, 1.0, 1.0));


SOFA_SCENECREATOR_API void addTetraFEM(simulation::NodeSPtr currentNode, const std::string& objectName,
                                       SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3);

SOFA_SCENECREATOR_API void addTriangleFEM(simulation::NodeSPtr currentNode, const std::string& objectName,
                                          SReal totalMass = 1.0, SReal young = 300, SReal poisson = 0.3);

/// Create a string composed of particles (at least 2) and springs
SOFA_SCENECREATOR_API simulation::NodeSPtr massSpringString(
        simulation::NodeSPtr parent,
        double x0, double y0, double z0, // start point,
        double x1, double y1, double z1, // end point
        unsigned numParticles,
        double totalMass,
        double stiffnessValue=1.0,
        double dampingRatio=0
        );

/// Helper class to create a component and add it as a child of a given Node
template<class T>
class addNew : public core::objectmodel::New<T>
{
    typedef typename T::SPtr SPtr;
public:
    addNew( simulation::NodeSPtr parent, const char* name="")
    {
        parent->addObject(*this);
        (*this)->setName(name);
    }
};

/// Initialize the sofa library and create the root of the scene graph
SOFA_SCENECREATOR_API simulation::NodeSPtr getRoot();

/// Initialize the sofa library and create the root of the scene graph
SOFA_SCENECREATOR_API simulation::NodeSPtr initSofa();

/// Initialize the scene graph
SOFA_SCENECREATOR_API void initScene(simulation::NodeSPtr root);

/// Clear the scene graph and return a pointer to the new root
SOFA_SCENECREATOR_API simulation::NodeSPtr clearScene();

/// Create a link from source to target.
SOFA_SCENECREATOR_API void setDataLink(core::objectmodel::BaseData* source, core::objectmodel::BaseData* target);

}// modeling

}// sofa

#endif
