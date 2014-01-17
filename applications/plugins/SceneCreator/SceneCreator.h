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
#include <sofa/component/mass/UniformMass.h>
#include <sofa/core/objectmodel/BaseData.h>

// Solvers
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

// Box roi
#include <sofa/component/engine/PairBoxRoi.h>
#include <sofa/component/engine/BoxROI.h>

// Constraint
#include <sofa/component/projectiveconstraintset/BilinearMovementConstraint.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

namespace sofa
{

/// BUGFIX: this SceneCreator class was renamed to SimpleSceneCreator,
/// in order to remove ambiguity with sofa::core::SceneCreator

class SOFA_SceneCreator_API SimpleSceneCreator
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


typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<SReal, Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;


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
SOFA_SceneCreator_API Vector getVector( core::ConstVecId id, bool independentOnly=true );

/** Initialize the sofa library and create the root of the scene graph
  */
SOFA_SceneCreator_API Node::SPtr initSofa();

/** Initialize the scene graph
  */
SOFA_SceneCreator_API void initScene();

/// Clear the scene graph and return a pointer to the new root
SOFA_SceneCreator_API simulation::Node::SPtr clearScene();

/// Create a link from source to target.  
SOFA_SceneCreator_API void setDataLink(core::objectmodel::BaseData* source, core::objectmodel::BaseData* target);

/// Structure which contains the nodes and the pointers useful for the patch test
template<class T>
struct PatchTestStruct
{
   simulation::Node::SPtr SquareNode;
   typename component::projectiveconstraintset::BilinearMovementConstraint<T>::SPtr bilinearConstraint;
   typename component::container::MechanicalObject<T>::SPtr dofs;
};

/// Create a scene with a regular grid and a bilinear constraint for patch test
template<class T> SOFA_SceneCreator_API PatchTestStruct<T> createRegularGridScene(simulation::Node::SPtr root ,Vec<3,SReal> startPoint, Vec<3,SReal> endPoint, unsigned numX, unsigned numY, unsigned numZ, helper::vector< Vec<6,SReal> > vecBoxRoi, Vec<6,SReal> inclusiveBox, Vec<6,SReal> includedBox);
template<class T> PatchTestStruct<T> createRegularGridScene(simulation::Node::SPtr root, Vec<3,SReal> startPoint, Vec<3,SReal> endPoint, unsigned numX, unsigned numY, unsigned numZ, helper::vector< Vec<6,SReal> > vecBoxRoi, Vec<6,SReal> inclusiveBox, Vec<6,SReal> includedBox)
{
    // Definitions
    PatchTestStruct<T> patchStruct;
    typedef typename component::container::MechanicalObject<T> MechanicalObject;
    typedef typename sofa::component::mass::UniformMass <T, SReal> UniformMass;
    typedef component::topology::RegularGridTopology RegularGridTopology;
    typedef typename component::engine::BoxROI<T> BoxRoi;
    typedef typename sofa::component::engine::PairBoxROI<T> PairBoxRoi;
    typedef typename component::projectiveconstraintset::BilinearMovementConstraint<T> BilinearMovementConstraint;
    typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

    // Root node
    root->setGravity( Coord3(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);

    // Node square
    simulation::Node::SPtr SquareNode = root->createChild("Square");
    
    // Euler implicit solver and cglinear solver
    component::odesolver::EulerImplicitSolver::SPtr solver = addNew<component::odesolver::EulerImplicitSolver>(SquareNode,"EulerImplicitSolver");
    solver->f_rayleighStiffness.setValue(0.5);
    solver->f_rayleighMass.setValue(0.5);
    CGLinearSolver::SPtr cgLinearSolver = addNew< CGLinearSolver >(SquareNode,"linearSolver");

    // Mass
    typename UniformMass::SPtr mass = addNew<UniformMass>(SquareNode,"mass");

    // Regular grid topology
    RegularGridTopology::SPtr gridMesh = addNew<RegularGridTopology>(SquareNode,"loader");
    gridMesh->setNumVertices(numX,numY,numZ);
    gridMesh->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    //Mechanical object
    patchStruct.dofs = addNew<MechanicalObject>(SquareNode,"mechanicalObject");
    patchStruct.dofs->setName("mechanicalObject");
    patchStruct.dofs->setSrc("@"+gridMesh->getName(), gridMesh.get());

    //BoxRoi to find all mesh points
    typename BoxRoi::SPtr boxRoi = addNew<BoxRoi>(SquareNode,"boxRoi");
    boxRoi->boxes.setValue(vecBoxRoi);

    //PairBoxRoi to define the constrained points = points of the border
    typename PairBoxRoi::SPtr pairBoxRoi = addNew<PairBoxRoi>(SquareNode,"pairBoxRoi");
    pairBoxRoi->inclusiveBox.setValue(inclusiveBox);
    pairBoxRoi->includedBox.setValue(includedBox);
      
    //Bilinear constraint 
    patchStruct.bilinearConstraint  = addNew<BilinearMovementConstraint>(SquareNode,"bilinearConstraint");
    setDataLink(&boxRoi->f_indices,&patchStruct.bilinearConstraint->m_meshIndices);
    setDataLink(&pairBoxRoi->f_indices,& patchStruct.bilinearConstraint->m_indices);
    setDataLink(&pairBoxRoi->f_pointsInROI,& patchStruct.bilinearConstraint->m_constrainedPoints);

    patchStruct.SquareNode = SquareNode;
    return patchStruct;
}
 

}// modeling

}// sofa

#endif
