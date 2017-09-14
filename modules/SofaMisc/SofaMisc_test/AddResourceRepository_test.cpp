#include <SofaTest/Sofa_test.h>
#include <sofa/helper/BackTrace.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>

#include <SofaMisc/AddResourceRepository.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace sofa
{

const std::string& START_STR("<Node name=\"root\"  >");
const std::string& END_STR("</Node>");

struct AddResourceRepository_test : public Sofa_test<>
{
    sofa::simulation::Node::SPtr m_root;
    std::string m_testRepoDir;

    void SetUp()
    {
        m_testRepoDir = std::string(MISC_TEST_RESOURCES_DIR) + std::string("/repo");
    }

    void buildScene(const std::string& repoPath)
    {
        std::string addRepoStr = "<AddResourceRepository path=\""+ repoPath + "\" />";

        std::string scene = START_STR + addRepoStr + END_STR;
        std::cout << scene << std::endl;

        m_root = sofa::simulation::SceneLoaderXML::loadFromMemory(
                "scene", scene.c_str(), scene.size());

        EXPECT_NE(m_root, nullptr);
    }

};


TEST_F(AddResourceRepository_test, RepoExists)
{
    std::string existFilename("somefilesomewhere.txt");
    std::string nopeFilename("somefilesomewherebutdoesnotexist.txt");

    EXPECT_MSG_EMIT(Error) ;

    EXPECT_FALSE(helper::system::DataRepository.findFile(existFilename));
    EXPECT_FALSE(helper::system::DataRepository.findFile(nopeFilename));

    buildScene(m_testRepoDir);

    EXPECT_FALSE(helper::system::DataRepository.findFile(nopeFilename));

    EXPECT_MSG_NOEMIT(Error);
    EXPECT_TRUE(helper::system::DataRepository.findFile(existFilename));

}

TEST_F(AddResourceRepository_test, RepoDoesNotExist)
{
    EXPECT_MSG_EMIT(Error) ;
    buildScene("/blabla/Repo_not_existing");
}


} // namespace sofa
