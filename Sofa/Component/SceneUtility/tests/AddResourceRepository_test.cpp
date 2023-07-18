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

#include <sofa/helper/BackTrace.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

#include <sofa/component/sceneutility/AddResourceRepository.h>

#include <sofa/simulation/graph/SimpleApi.h>

namespace sofa
{

const std::string& START_STR("<Node name=\"root\"  >");
const std::string& END_STR("</Node>");

struct AddResourceRepository_test : public BaseSimulationTest
{
    sofa::simulation::Node::SPtr m_root;
    std::string m_testRepoDir;

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.SceneUtility");

        m_testRepoDir = std::string(SOFA_COMPONENT_SCENEUTILITY_TEST_RESOURCES_DIR) + std::string("/repo");
    }

    void buildScene(const std::string& repoType, const std::string& repoPath)
    {
        const std::string addRepoStr = "<" + repoType + " path=\""+ repoPath + "\" />";

        const std::string scene = START_STR + addRepoStr + END_STR;
        std::cout << scene << std::endl;

        m_root = sofa::simulation::SceneLoaderXML::loadFromMemory("scene", scene.c_str());

        EXPECT_NE(m_root, nullptr);
    }
};


TEST_F(AddResourceRepository_test, AddDataRepository_RepoExists)
{
    std::string existFilename("somefilesomewhere.txt");
    std::string nopeFilename("somefilesomewherebutdoesnotexist.txt");

    EXPECT_MSG_EMIT(Error) ;

    EXPECT_FALSE(helper::system::DataRepository.findFile(existFilename));
    EXPECT_FALSE(helper::system::DataRepository.findFile(nopeFilename));

    buildScene("AddDataRepository", m_testRepoDir);

    EXPECT_FALSE(helper::system::DataRepository.findFile(nopeFilename));

    EXPECT_MSG_NOEMIT(Error);
    EXPECT_TRUE(helper::system::DataRepository.findFile(existFilename));
}

TEST_F(AddResourceRepository_test, AddDataRepository_RepoDoesNotExist)
{
    EXPECT_MSG_EMIT(Error) ;
    buildScene("AddDataRepository", "/blabla/Repo_not_existing");
}


} // namespace sofa
