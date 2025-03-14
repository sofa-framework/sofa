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

namespace sofa {

using namespace type;
using namespace testing;
using namespace defaulttype;
using core::objectmodel::New;

/// Create a stiff string
Node::SPtr particleUnderGravity(Node::SPtr parent,
                                std::string nodeName,
                                Vec3d gravity,
                                bool useGravity)
{
    Node::SPtr node = simpleapi::createChild(parent, nodeName);

    if(useGravity)
    {
        Node::SPtr node = simpleapi::createChild(parent, nodeName, { {"gravity", simpleapi::str(gravity)} });
    }
    else
    {
        Node::SPtr node = simpleapi::createChild(parent, nodeName);
    }

    Vec3d startPoint(0., 0., 0.);

    simpleapi::createObject(node, "EulerImplicitSolver", {});
    simpleapi::createObject(node, "CGLinearSolver", {
        { "iterations", simpleapi::str(25)},
        { "tolerance", simpleapi::str(1e-5)},
        { "threshold", simpleapi::str(1e-5)},
    });

    simpleapi::createObject(node, "MechanicalObject", {
                                {"name", nodeName + "_DOF"},
                                {"position", simpleapi::str(startPoint)}
        });

    simpleapi::createObject(node, "UniformMass", {
                                {"name", nodeName + "_mass"},
                                {"totalMass", simpleapi::str(1.)} });

    return node;
}

/// Test for the gravity context data
struct Gravity_set_at_root_node : public BaseSimulationTest, NumericTest<SReal>
{
    Gravity_set_at_root_node()
    {
        Vec3d gravity1(0., 10., 0.);
        Vec3d gravity2(0., 20., 0.);
        Vec3d gravity3(0., 30., 0.);

        //*******
        const auto simu = simpleapi::createSimulation();
        const simulation::Node::SPtr root = simpleapi::createRootNode(simu, "root", { {"gravity", simpleapi::str(gravity1)} });
        //*******
        // load appropriate modules
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");

        // avoid warnings
        simpleapi::createObject(root, "DefaultAnimationLoop", {});
        simpleapi::createObject(root, "DefaultVisualManagerLoop", {});

        //*********
        // create scene
        simulation::Node::SPtr particule1 = particleUnderGravity(root, "Node-no-gravity-set", type::Vec3(), false);
        Node::SPtr emptyNode1 = simpleapi::createChild(root, "EmptyNode1");
        simulation::Node::SPtr particule2 = particleUnderGravity(emptyNode1, "Node-no-gravity-set", type::Vec3(), false);

        simulation::Node::SPtr particule3 = particleUnderGravity(root, "Node-gravity-set", gravity2, true);
        Node::SPtr emptyNode2 = simpleapi::createChild(root, "EmptyNode2");
        simulation::Node::SPtr particule4 = particleUnderGravity(emptyNode2, "Node-gravity-set", gravity2, true);

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********
        // run simulation

        // do one time step
        sofa::simulation::node::animate(root.get(), 1_sreal);

        EXPECT_EQ(gravity1,particule1->getGravity());
        EXPECT_EQ(gravity1,particule2->getGravity());
        EXPECT_EQ(gravity2,particule3->getGravity());
        EXPECT_EQ(gravity2,particule4->getGravity());

        root->setGravity(gravity3);
        emptyNode2->setGravity(gravity3);

        // do another time step
        sofa::simulation::node::animate(root.get(), 1_sreal);

        EXPECT_EQ(gravity3,particule1->getGravity());
        EXPECT_EQ(gravity3,particule2->getGravity());
        EXPECT_EQ(gravity2,particule3->getGravity());
        EXPECT_EQ(gravity3,particule4->getGravity());
    }
};


struct Gravity_not_set_at_root_node  : public BaseSimulationTest, NumericTest<SReal>
{

    Gravity_not_set_at_root_node()
    {
        Vec3d default_gravity(0., -9.81, 0.);
        Vec3d gravity2(0., 20., 0.);
        Vec3d gravity3(0., 30., 0.);

        //*******
        const auto simu = simpleapi::createSimulation();
        const simulation::Node::SPtr root = simpleapi::createRootNode(simu, "root");
        //*******
        // load appropriate modules
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Iterative");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Mass");

        // avoid warnings
        simpleapi::createObject(root, "DefaultAnimationLoop", {});
        simpleapi::createObject(root, "DefaultVisualManagerLoop", {});

        //*********
        // create scene
        simulation::Node::SPtr particule1 = particleUnderGravity(root, "Node-no-gravity-set", type::Vec3(), false);
        Node::SPtr emptyNode1 = simpleapi::createChild(root, "EmptyNode1");
        simulation::Node::SPtr particule2 = particleUnderGravity(emptyNode1, "Node-no-gravity-set", type::Vec3(), false);

        simulation::Node::SPtr particule3 = particleUnderGravity(root, "Node-gravity-set", gravity2, true);
        Node::SPtr emptyNode2 = simpleapi::createChild(root, "EmptyNode2");
        simulation::Node::SPtr particule4 = particleUnderGravity(emptyNode2, "Node-gravity-set", gravity2, true);

        // end create scene
        //*********
        sofa::simulation::node::initRoot(root.get());
        //*********
        // run simulation

        // do one time step
        sofa::simulation::node::animate(root.get(), 1_sreal);

        EXPECT_EQ(default_gravity,particule1->getGravity());
        EXPECT_EQ(default_gravity,particule2->getGravity());
        EXPECT_EQ(gravity2,particule3->getGravity());
        EXPECT_EQ(gravity2,particule4->getGravity());

        root->setGravity(gravity3);
        emptyNode2->setGravity(gravity3);

        // do another time step
        sofa::simulation::node::animate(root.get(), 1_sreal);

        EXPECT_EQ(gravity3,particule1->getGravity());
        EXPECT_EQ(gravity3,particule2->getGravity());
        EXPECT_EQ(gravity2,particule3->getGravity());
        EXPECT_EQ(gravity3,particule4->getGravity());
    }

};

TEST_F( Gravity_not_set_at_root_node, check ){}
TEST_F( Gravity_set_at_root_node, check ){}

}// namespace sofa







