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

#include <sofa/simulation/graph/SimpleApi.h>

#include <sofa/testing/NumericTest.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::odesolver::testing
{

/// Create a mass spring system
inline simulation::Node::SPtr createMassSpringSystem(
    simulation::Node::SPtr root,
    const std::string& stiffness,
    const std::string& mass,
    const std::string& restLength,
    const std::string& xFixedPoint,
    const std::string& vFixedPoint,
    const std::string& xMass,
    const std::string& vMass)
{
    // Fixed point
    const auto fixedPointNode = simpleapi::createChild(root, "FixedPointNode" );

    simpleapi::createObject(fixedPointNode, "MechanicalObject", {
        { "name","fixedPoint"},
        { "template","Vec3"},
        { "position", xFixedPoint},
        { "velocity", vFixedPoint},
    });

    simpleapi::createObject(fixedPointNode, "FixedProjectiveConstraint", {
        { "name","fixed"},
        { "indices", "0"},
    });

    // Mass
    const auto massNode = simpleapi::createChild(root, "MassNode");

    simpleapi::createObject(massNode, "MechanicalObject", {
        { "name","massDof"},
        { "template","Vec3"},
        { "position", xMass},
        { "velocity", vMass}
    });

    simpleapi::createObject(massNode, "UniformMass", {
        { "name","mass"},
        { "totalMass", mass}
    });

    std::ostringstream oss;
    oss << 0 << " " << 0 << " " << stiffness << " " << 0 << " " << restLength;

    // attach a spring
    simpleapi::createObject(root, "StiffSpringForceField", {
        { "name","ff"},
        { "spring", oss.str()},
        { "object1", "@FixedPointNode/fixedPoint"},
        { "object2", "@MassNode/massDof"},
    });

    return root;
}

template<typename DataTypes>
inline simulation::Node::SPtr createMassSpringSystem(
    simulation::Node::SPtr root,
    double stiffness,
    double mass,
    double restLength,
    typename DataTypes::VecCoord xFixedPoint,
    typename DataTypes::VecDeriv vFixedPoint,
    typename DataTypes::VecCoord xMass,
    typename DataTypes::VecDeriv vMass)
{
    return createMassSpringSystem(root,
        simpleapi::str(stiffness), simpleapi::str(mass), simpleapi::str(restLength),
        simpleapi::str(xFixedPoint), simpleapi::str(vFixedPoint),
        simpleapi::str(xMass), simpleapi::str(vMass)
    );
}

} // namespace sofa::component::odesolver::testing
