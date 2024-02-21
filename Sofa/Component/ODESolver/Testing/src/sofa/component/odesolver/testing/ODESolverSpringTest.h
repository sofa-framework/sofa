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

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/testing/NumericTest.h>
#include <sofa/simulation/Node.h>

#include <sofa/component/odesolver/testing/MassSpringSystemCreation.h>

namespace sofa::component::odesolver::testing
{

struct ODESolverSpringTest : public BaseSimulationTest
{
    SceneInstance m_si{};

    inline void prepareScene(double K, double m, double l0)
    {
        // Create the scene
        m_si.root->setGravity({ 0, -10, 0 });

        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::importPlugin("Sofa.Component.Constraint.Projective");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");

        // remove warnings
        simpleapi::createObject(m_si.root, "DefaultAnimationLoop", {});
        simpleapi::createObject(m_si.root, "DefaultVisualManagerLoop", {});
        
        simpleapi::createObject(m_si.root, "CGLinearSolver", {
            { "iterations", simpleapi::str(3000)},
            { "tolerance", simpleapi::str(1e-12)},
            { "threshold", simpleapi::str(1e-12)},
            });

        // Add mass spring system
        createMassSpringSystem(
            m_si.root,   // add mass spring system to the node containing solver
            simpleapi::str(K),      // stiffness
            simpleapi::str(m),      // mass
            simpleapi::str(l0),     // spring rest length
            std::string("0.0 2.0 0.0"), // Initial position of fixed point
            std::string("0.0 0.0 0.0"), // Initial velocity of fixed point
            std::string("0.0 1.0 0.0"),  // Initial position of mass
            std::string("0.0 0.0 0.0")); // Initial velocity of mass
    }
};

} // namespace sofa::component::odesolver::testing
