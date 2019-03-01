/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest;

#include <SofaSimulationGraph/SimpleApi.h>
using sofa::simulation::Node;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3;

class ReadState_test : public BaseSimulationTest
{
public:
    /// Run seven steps of simulation then check results
    bool testDefaultBehavior()
    {
        double dt = 0.01;
        sofa::simpleapi::importPlugin("SofaAllCommonComponents") ;
        auto simulation = sofa::simpleapi::createSimulation();
        Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");

        /// no need of gravity, the file .data is just read
        root->setGravity(Vec3(0.0,0.0,0.0));
        root->setDt(dt);

        Node::SPtr childNode = sofa::simpleapi::createChild(root, "Particle");

        auto meca = sofa::simpleapi::createObject(childNode, "MechanicalObject",
                                                  {{"size", "1"}});

        sofa::simpleapi::createObject(childNode, "ReadState",
                                      {{"filename", std::string(SOFAGENERALLOADER_TESTFILES_DIR)+"particleGravityX.data"}});

        simulation->init(root.get());
        for(int i=0; i<7; i++)
        {
            simulation->animate(root.get(), dt);
        }

        EXPECT_EQ(meca->findData("position")->getValueString(),
                  std::string("0 0 -0.017658"));
        return true;
    }

    /// Run seven steps of simulation then check results
    bool testLoadFailure()
    {
        sofa::simpleapi::importPlugin("SofaAllCommonComponents") ;
        auto simulation = sofa::simpleapi::createSimulation();
        Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");

        auto meca = sofa::simpleapi::createObject(root, "MechanicalObject",
                                                  {{"size", "1"}});

        {
            EXPECT_MSG_EMIT(Error);
            sofa::simpleapi::createObject(root, "ReadState",
                                      {{"filename", std::string(SOFAGENERALLOADER_TESTFILES_DIR)+"invalidFile.txt"}});
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
