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
#include <gtest/gtest.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/fwd.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/testing/TestMessageHandler.h>

namespace sofa
{

TEST(simulationLoad, existingFilename)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;
    EXPECT_MSG_NOEMIT(Error);

    constexpr std::string_view filename { "Demos/caduceus.scn" };
    const std::string path = helper::system::DataRepository.getFile(std::string{filename});

    simulation::Simulation* simulation = sofa::simulation::getSimulation();
    ASSERT_NE(simulation, nullptr);
    const simulation::Node::SPtr groot = sofa::simulation::node::load(path, false, {});
    EXPECT_NE(groot, nullptr);
    sofa::simulation::node::unload(groot);
}


TEST(simulationLoad, nonExistingFilename)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;
    EXPECT_MSG_EMIT(Error);

    constexpr std::string_view filename { "aFileThatDoesNotExist.scn" };

    simulation::Simulation* simulation = simulation::getSimulation();
    ASSERT_NE(simulation, nullptr);
    const simulation::Node::SPtr groot = sofa::simulation::node::load(std::string{filename}, false, {});
    EXPECT_EQ(groot, nullptr);
    sofa::simulation::node::unload(groot);
}

}
