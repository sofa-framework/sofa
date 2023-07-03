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

#include <sofa/simulation/graph/SimpleApi.h>
using sofa::simulation::Node;

#include <sofa/type/Vec.h>
using sofa::type::Vec3;

class PolynomialRestShapeSpringsForceField_test : public BaseSimulationTest
{
public:
    /// Run one stepsof simulation then check results
    bool testSpringForce()
    {
        const double dt = 0.01;
        const auto simulation = sofa::simpleapi::createSimulation();
        const Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");

        /// no need of gravity, the file .data is just read
        root->setGravity(Vec3(0.0,0.0,0.0));
        root->setDt(dt);

        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Forward");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");

        const Node::SPtr childNode = sofa::simpleapi::createChild(root, "Particle");
        sofa::simpleapi::createObject(childNode, "EulerExplicitSolver");
        const auto meca = sofa::simpleapi::createObject(childNode, "MechanicalObject", {{"rest_position", "0 0 0"},{"position", "1.1 0 0"}});
        sofa::simpleapi::createObject(childNode, "EulerExplicitSolver", {{"totalMass", "1.0"}});

        // Add the spring to test
        sofa::simpleapi::createObject(childNode, "PolynomialRestShapeSpringsForceField", {{"polynomialStiffness", "10 10"},{"polynomialDegree", "2"},{"points", "0"},{"smoothShift", "1e-4"},{"smoothScale", "1e7"}});

        sofa::simulation::node::initRoot(root.get());
        for(int i=0; i<2; i++)
        {
            sofa::simulation::node::animate(root.get(), dt);
        }

        EXPECT_EQ(meca->findData("force")->getValueString(), std::string("-23.1 0 0")); //F = S sigma(L) where L = 1.1 , S=1 and sigma(L) = 10*L + 10*L^2
        return true;
    }
};


/// Test : forces returned by the forcefield
TEST_F(PolynomialRestShapeSpringsForceField_test , test_springForce)
{
    ASSERT_TRUE( this->testSpringForce() );
}
