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

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa {

using namespace type;
using namespace testing;
using namespace defaulttype;
using core::objectmodel::New;
using namespace sofa::component::statecontainer;


/// Test for the gravity context data
struct EulerImplicit_with_damping_forcefield : public BaseSimulationTest, NumericTest<SReal>
{
    EulerImplicit_with_damping_forcefield()
    {
        //*******
        const auto simu = simpleapi::createSimulation();
        const simulation::Node::SPtr root = simpleapi::createRootNode(simu, "root");

        Vec3d zeroVec3(0., 0., 0.);
        Vec3d oneVec3(1., 1., 1.);
        Vec3d expectedPosition(10., 10., 10.);
        Vec3d threshold(0.1, 0.1, 0.1);
        root->setGravity(zeroVec3);

        //*******
        // load appropriate modules
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::importPlugin("Sofa.Component.MechanicalLoad");

        // avoid warnings
        simpleapi::createObject(root, "DefaultAnimationLoop", {});
        simpleapi::createObject(root, "DefaultVisualManagerLoop", {});

        //*********
        // create scene
        Node::SPtr dampedParticule = simpleapi::createChild(root, "dampedParticule");

        simpleapi::createObject(dampedParticule, "EulerImplicitSolver", {{ "rayleighStiffness", simpleapi::str(0.)},
                                                                         { "rayleighMass", simpleapi::str(0.)}, });
        simpleapi::createObject(dampedParticule, "CGLinearSolver", {
            { "iterations", simpleapi::str(25)},
            { "tolerance", simpleapi::str(1e-15)},
            { "threshold", simpleapi::str(1e-15)},
            });
        simpleapi::createObject(dampedParticule, "MechanicalObject", {
            { "template", simpleapi::str("Vec3")},
            { "position", simpleapi::str(zeroVec3)},
            { "velocity", simpleapi::str(oneVec3)},
            { "force", simpleapi::str(zeroVec3)},
            { "externalForce", simpleapi::str(zeroVec3)},
            { "derivX", simpleapi::str(zeroVec3)},
            });
        simpleapi::createObject(dampedParticule, "UniformMass", {
            { "totalMass", simpleapi::str(1.0)},
            });
        simpleapi::createObject(dampedParticule, "UniformVelocityDampingForceField", {
            { "template", simpleapi::str("Vec3")},
            });

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********
        // run simulation

        // compute 2500 time steps
        for(int i=0; i<2500; i++)
            sofa::simulation::node::animate(root.get(), 0.02);

        // access the MechanicalObect (access position of the dampedParticule)
        typename MechanicalObject<sofa::defaulttype::Vec3dTypes>::SPtr dofs = dampedParticule->get<MechanicalObject<sofa::defaulttype::Vec3dTypes>>(root->SearchDown);
        sofa::defaulttype::Vec3dTypes::Coord position = dofs.get()->read(sofa::core::ConstVecCoordId::position())->getValue()[0];

        // save it as Vec3d for comparison with expected result
        Vec3d finalPosition(position[0], position[1], position[2]);

        // Position at t∞ is (10,10,10), see https://github.com/sofa-framework/sofa/pull/4848#issuecomment-2263947900
        EXPECT_LT(expectedPosition-finalPosition,threshold);
    }
};



TEST_F( EulerImplicit_with_damping_forcefield, check ){}

}// namespace sofa







