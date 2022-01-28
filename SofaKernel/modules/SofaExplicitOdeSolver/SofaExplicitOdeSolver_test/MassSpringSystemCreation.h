/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <SceneCreator/SceneCreator.h>

#include <sofa/testing/NumericTest.h>
#include <sofa/simulation/Node.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaDeformable/StiffSpringForceField.h>

namespace sofa
{

/// Create a mass srping system
template<typename DataTypes>
simulation::Node::SPtr createMassSpringSystem(
    simulation::Node::SPtr root,
    double stiffness,
    double mass,
    double restLength,
    typename DataTypes::VecCoord xFixedPoint,
    typename DataTypes::VecDeriv vFixedPoint,
    typename DataTypes::VecCoord xMass,
    typename DataTypes::VecDeriv vMass)
{

    typedef component::container::MechanicalObject<defaulttype::Vec3Types> MechanicalObject3;
    typedef component::projectiveconstraintset::FixedConstraint<defaulttype::Vec3Types> FixedConstraint3;
    typedef component::mass::UniformMass<defaulttype::Vec3Types> UniformMass3;
    typedef component::interactionforcefield::StiffSpringForceField<defaulttype::Vec3Types > StiffSpringForceField3;

    // Fixed point
    simulation::Node::SPtr fixedPointNode = root->createChild("FixedPointNode");
    MechanicalObject3::SPtr FixedPoint = modeling::addNew<MechanicalObject3>(fixedPointNode, "fixedPoint");

    // Set position and velocity
    FixedPoint->resize(1);
    MechanicalObject3::WriteVecCoord xdof = FixedPoint->writePositions();
    sofa::testing::copyToData(xdof, xFixedPoint);
    MechanicalObject3::WriteVecDeriv vdof = FixedPoint->writeVelocities();
    sofa::testing::copyToData(vdof, vFixedPoint);

    FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(fixedPointNode, "FixedPointNode");
    fixed->addConstraint(0);      // attach particle


    // Mass
    simulation::Node::SPtr massNode = root->createChild("MassNode");
    MechanicalObject3::SPtr massDof = modeling::addNew<MechanicalObject3>(massNode, "massNode");

    // Set position and velocity
    FixedPoint->resize(1);
    MechanicalObject3::WriteVecCoord xMassDof = massDof->writePositions();
    sofa::testing::copyToData(xMassDof, xMass);
    MechanicalObject3::WriteVecDeriv vMassDof = massDof->writeVelocities();
    sofa::testing::copyToData(vMassDof, vMass);

    UniformMass3::SPtr massPtr = modeling::addNew<UniformMass3>(massNode, "mass");
    massPtr->d_totalMass.setValue(mass);

    // attach a spring
    StiffSpringForceField3::SPtr spring = core::objectmodel::New<StiffSpringForceField3>(FixedPoint.get(), massDof.get());
    root->addObject(spring);
    spring->addSpring(0, 0, stiffness, 0, restLength);

    return root;
}

} // namespace sofa