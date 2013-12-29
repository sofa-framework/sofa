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

#ifndef SOFA_SIMPLEOBJECTCREATOR_H
#define SOFA_SIMPLEOBJECTCREATOR_H

#include "initSceneCreator.h"
#include <string>
#include <sofa/simulation/common/Node.h>
#include <sofa/component/loader/MeshObjLoader.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

#ifdef SOFA_BUILD_OBJECTCREATOR
#	define SOFA_OBJECTCREATOR_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_OBJECTCREATOR_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace sofa
{

/// BUGFIX: this SceneCreator class was renamed to SimpleSceneCreator,
/// in order to remove ambiguity with sofa::core::SceneCreator

class SOFA_OBJECTCREATOR_API SimpleSceneCreator
{
public:

    typedef SReal Scalar;
    typedef Vec<3,SReal> Vec3;
    typedef Vec<1,SReal> Vec1;

    static simulation::Node::SPtr CreateRootWithCollisionPipeline(const std::string &responseType=std::string("default"));
    static simulation::Node::SPtr CreateEulerSolverNode(simulation::Node::SPtr parent, const std::string& name, const std::string &integrationScheme=std::string("Implicit"));


    static simulation::Node::SPtr CreateObstacle(simulation::Node::SPtr parent, const std::string &filenameCollision, const std::string filenameVisual, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());

    //Create a collision node using Barycentric Mapping, using a 3d model specified by filename.
    //elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
    //an initial transformation can be performed
    static simulation::Node::SPtr CreateCollisionNodeVec3(simulation::Node::SPtr parent, MechanicalObject3::SPtr dof, const std::string &filename, const std::vector<std::string> &elements,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());
    static simulation::Node::SPtr CreateVisualNodeVec3(simulation::Node::SPtr parent, MechanicalObject3::SPtr dof,  const std::string &filename, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());



    //Create a collision node using Rigid Mapping, using a 3d model specified by filename.
    //elements is a vector of type of collision models (Triangle, Line, Point, Sphere)
    //an initial transformation can be performed
    static simulation::Node::SPtr CreateCollisionNodeRigid(simulation::Node::SPtr parent, MechanicalObjectRigid3::SPtr dofRigid,  const std::string &filename, const std::vector<std::string> &elements,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());
    static simulation::Node::SPtr CreateVisualNodeRigid(simulation::Node::SPtr parent, MechanicalObjectRigid3::SPtr  dofRigid,  const std::string &filename, const std::string& color,
            const Deriv3& translation=Deriv3(), const Deriv3 &rotation=Deriv3());

    static simulation::Node::SPtr createGridScene(Vec3 startPoint, Vec3 endPoint, unsigned numX, unsigned numY, unsigned numZ, double totalMass, double stiffnessValue=1.0, double dampingRatio=0 );


private:
    static void AddCollisionModels(simulation::Node::SPtr CollisionNode, const std::vector<std::string> &elements);
};

namespace modeling {

using namespace simulation;

/// Dense state vector, used externally
typedef Eigen::VectorXd Vector;

/// Dense state vector deriving from BaseVector, used to access data in the scene graph
typedef component::linearsolver::FullVector<SReal> FullVector;



/** Create a string composed of particles (at least 2) and springs */
SOFA_SceneCreator_API Node::SPtr massSpringString(
        simulation::Node::SPtr parent,
        double x0, double y0, double z0, // start point,
        double x1, double y1, double z1, // end point
        unsigned numParticles,
        double totalMass,
        double stiffnessValue=1.0,
        double dampingRatio=0
        );


/** Helper class to create a component and add it as a child of a given Node */
template<class T>
class addNew : public objectmodel::New<T>
{
    typedef typename T::SPtr SPtr;
public:
    addNew( Node::SPtr parent, const char* name="")
    {
        parent->addObject(*this);
        (*this)->setName(name);
    }

};


SOFA_SceneCreator_API Node::SPtr getRoot();

/// Get a state vector from the scene graph. Includes only the independent state values, or also the mapped ones, depending on the flag.
Vector assembled( core::ConstVecId id, bool independentOnly=true );

/** Initialize the sofa library and create the root of the scene graph
  */
SOFA_SceneCreator_API Node::SPtr initSofa();

/** Initialize the scene graph
  */
SOFA_SceneCreator_API void initScene();




}// modeling

}// sofa

#endif
