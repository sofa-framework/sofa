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

class ReadState_test : public BaseSimulationTest
{
public:
    /// Run seven steps of simulation then check results
    bool testDefaultBehavior()
    {
        const double dt = 0.01;
        const auto simulation = sofa::simpleapi::createSimulation();
        const Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");
        sofa::simpleapi::createObject(root, "RequiredPlugin", { { "name","Sofa.Component.Playback" } });
        sofa::simpleapi::createObject(root, "RequiredPlugin", { { "name","Sofa.Component.StateContainer" } });

        /// no need of gravity, the file .data is just read
        root->setGravity(Vec3(0.0,0.0,0.0));
        root->setDt(dt);

        const Node::SPtr childNode = sofa::simpleapi::createChild(root, "Particle");

        const auto meca = sofa::simpleapi::createObject(childNode, "MechanicalObject",
                                                        {{"size", "1"}});

        sofa::simpleapi::createObject(childNode, "ReadState",
                                      {{"filename", std::string(SOFA_COMPONENT_PLAYBACK_TEST_FILES_DIR)+"particleGravityX.data"}});

        sofa::simulation::node::initRoot(root.get());
        for(int i=0; i<7; i++)
        {
            sofa::simulation::node::animate(root.get(), dt);
        }

        EXPECT_EQ(meca->findData("position")->getValueString(),
                  std::string("0 0 -0.017658"));
        return true;
    }

    /// Run seven steps of simulation then check results
    bool testLoadFailure()
    {
        const auto simulation = sofa::simpleapi::createSimulation();
        const Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");
        sofa::simpleapi::createObject(root, "RequiredPlugin", { { "name","Sofa.Component.Playback" } });
        sofa::simpleapi::createObject(root, "RequiredPlugin", { { "name","Sofa.Component.StateContainer" } });

        auto meca = sofa::simpleapi::createObject(root, "MechanicalObject",
                                                  {{"size", "1"}});

        {
            EXPECT_MSG_EMIT(Error);
            sofa::simpleapi::createObject(root, "ReadState",
                                      {{"filename", std::string(SOFA_COMPONENT_PLAYBACK_TEST_FILES_DIR)+"invalidFile.txt"}});
            sofa::simulation::node::initRoot(root.get());
        }

        return true;
    }
};

/// Test : read positions of a particle falling under gravity
TEST_F(ReadState_test , test_defaultBehavior)
{
    ASSERT_TRUE( this->testDefaultBehavior() );
}

/// Test : when happens when unable to load the file ?
TEST_F(ReadState_test , test_loadFailure)
{
    ASSERT_TRUE( this->testLoadFailure() );
}
