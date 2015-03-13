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
#ifndef SOFA_Elasticity_test_H
#define SOFA_Elasticity_test_H


#include "Sofa_test.h"
#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/forcefield/TetrahedralTensorMassForceField.h>
#include <sofa/component/forcefield/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/topology/TopologySparseData.inl>
#include <sofa/component/forcefield/TrianglePressureForceField.h>
#include <sofa/component/projectiveconstraintset/AffineMovementConstraint.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/engine/PairBoxRoi.h>
#include <sofa/component/engine/GenerateCylinder.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.h>

namespace sofa {


using namespace simulation;
using namespace modeling;

/// Structure which contains the nodes and the pointers useful for the patch test
template<class T>
struct  PatchTestStruct
{
   simulation::Node::SPtr SquareNode;
   typename component::projectiveconstraintset::AffineMovementConstraint<T>::SPtr affineConstraint;
   typename component::container::MechanicalObject<T>::SPtr dofs;
};

/// Structure which contains the nodes and the pointers useful for the patch test
template<class T>
struct   CylinderTractionStruct
{
   simulation::Node::SPtr root;
   typename component::container::MechanicalObject<T>::SPtr dofs;
   typename component::forcefield::TrianglePressureForceField<T>::SPtr forceField;
};


template< class DataTypes>
 struct SOFA_TestPlugin_API Elasticity_test: public Sofa_test<typename DataTypes::Real>
{
    typedef component::container::MechanicalObject<DataTypes> DOFs;
    typedef typename DOFs::Real  Real;
    typedef typename DOFs::Coord  Coord;
    typedef typename DOFs::Deriv  Deriv;
    typedef typename DOFs::VecCoord  VecCoord;
    typedef typename DOFs::VecDeriv  VecDeriv;
    typedef typename DOFs::ReadVecCoord  ReadVecCoord;
    typedef typename DOFs::WriteVecCoord WriteVecCoord;
    typedef typename DOFs::ReadVecDeriv  ReadVecDeriv;
    typedef typename DOFs::WriteVecDeriv WriteVecDeriv;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;


/// Create a scene with a regular grid and an affine constraint for patch test

PatchTestStruct<DataTypes> createRegularGridScene(
        Node::SPtr root,
        Coord startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        Vec<6,SReal> entireBoxRoi,
        Vec<6,SReal> inclusiveBox,
        Vec<6,SReal> includedBox);

CylinderTractionStruct<DataTypes>  createCylinderTractionScene(
        int resolutionCircumferential,
        int resolutionRadial,
        int resolutionHeight,
        int maxIter);


/// Create an assembly of a siff hexahedral grid with other objects
simulation::Node::SPtr createGridScene(
        Coord startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        SReal totalMass,
        SReal stiffnessValue,
        SReal dampingRatio );

/// Create a mass srping system
simulation::Node::SPtr createMassSpringSystem(
        simulation::Node::SPtr root,
        double stiffness,
        double mass,
        double restLength,
        VecCoord xFixedPoint,
        VecDeriv vFixedPoint,
        VecCoord xMass,
        VecDeriv vMass);

/// Create sun-planet system
simulation::Node::SPtr createSunPlanetSystem(
        simulation::Node::SPtr root,
        double mSun,
        double mPlanet,
        double g,
        Coord xSun,
        Deriv vSun,
        Coord xPlanet,
        Deriv vPlanet);

};




} // namespace sofa

#endif
