/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

class MeshXspLoader_test : public BaseSimulationTest
{
public:
    /// Run seven steps of simulation then check results
    bool testDefaultBehavior()
    {
        sofa::simpleapi::importPlugin("SofaAllCommonComponents") ;
        auto simulation = sofa::simpleapi::createSimulation();
        Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");

        auto loader = sofa::simpleapi::createObject(root, "MeshXspLoader",
                                      {{"filename", std::string(SOFAGENERALLOADER_TESTFILES_DIR)+"test.xs3"}});
        simulation->init(root.get());

        return true;
    }

    /// Run seven steps of simulation then check results
    bool testInvalidFile()
    {
        sofa::simpleapi::importPlugin("SofaAllCommonComponents") ;
        auto simulation = sofa::simpleapi::createSimulation();
        Node::SPtr root = sofa::simpleapi::createRootNode(simulation, "root");

        {
            EXPECT_MSG_EMIT(Error);
            sofa::simpleapi::createObject(root, "MeshXspLoader",
                                      {{"filename", std::string(SOFAGENERALLOADER_TESTFILES_DIR)+"invalidFile.xs3"}});
            simulation->init(root.get());
        }

        return true;
    }
};

// Test : read positions of a particle falling under gravity
TEST_F(MeshXspLoader_test , test_defaultBehavior)
{
    ASSERT_TRUE( this->testDefaultBehavior() );
}

// Test : read positions of a particle falling under gravity
TEST_F(MeshXspLoader_test , test_invalidFile)
{
    ASSERT_TRUE( this->testInvalidFile() );
}
