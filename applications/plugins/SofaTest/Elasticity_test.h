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
#ifndef SOFA_Elasticity_test_H
#define SOFA_Elasticity_test_H


#include "Sofa_test.h"
#include <SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/mechanicalload/TrianglePressureForceField.h>
#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.h>
#include <sofa/component/statecontainer/MechanicalObject.h>

namespace sofa {

/// Structure which contains the nodes and the pointers useful for the patch test
template<class T>
struct PatchTestStruct
{
    simulation::Node::SPtr SquareNode;
    typename component::constraint::projective::AffineMovementProjectiveConstraint<T>::SPtr affineConstraint;
    typename component::statecontainer::MechanicalObject<T>::SPtr dofs;
};

/// Structure which contains the nodes and the pointers useful for the patch test
template<class T>
struct CylinderTractionStruct
{
    simulation::Node::SPtr root;
    typename component::statecontainer::MechanicalObject<T>::SPtr dofs;
    typename component::mechanicalload::TrianglePressureForceField<T>::SPtr forceField;
};


template< class DataTypes>
struct SOFA_SOFATEST_API Elasticity_test: public Sofa_test<typename DataTypes::Real>
{
    typedef component::statecontainer::MechanicalObject<DataTypes> DOFs;
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
        simulation::Node::SPtr root,
        Coord startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        sofa::type::Vec<6,SReal> entireBoxRoi,
        sofa::type::Vec<6,SReal> inclusiveBox,
        sofa::type::Vec<6,SReal> includedBox);

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

};




} // namespace sofa

#endif
