/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_Regression_test_H
#define SOFA_Regression_test_H

#include <sofa/helper/testing/BaseTest.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaExporter/WriteState.h>
#include <SofaGeneralLoader/ReadState.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaValidation/CompareState.h>

#include <SofaTest/Sofa_test.h>
#include <sofa/helper/system/FileRepository.h>

using sofa::helper::testing::BaseTest;

namespace sofa 
{

/// a struct to store all info to perform the regression test
struct RegressionTestData
{
    RegressionTestData(const std::string& fileScenePath, const std::string& fileRefPath, unsigned int steps, double epsilon)
        : fileScenePath(fileScenePath)
        , fileRefPath(fileRefPath)
        , steps(steps)
        , epsilon(epsilon)
    {}

    std::string fileScenePath;
    std::string fileRefPath;
    unsigned int steps;
    double epsilon;
};


/// To Perform a Regression Test on scenes
///
/// A scene is run for a given number of steps and the state (position/velocity) of every independent dofs is stored in files. These files must be added to the repository.
/// At each commit, a test runs the scenes again for the same given number of steps. Then the independent states are compared to the references stored in the files.
///
/// The reference files are generated when running the test for the first time on a scene.
/// @warning These newly created reference files must be added to the repository.
/// If the result of the simulation changed voluntarily, these files must be manually deleted (locally) so they can be created again (by running the test).
/// Their modifications must be pushed to the repository.
///
/// Scene tested for regression must be listed in a file "list.txt" located in a "regression" directory in the test directory ( e.g. myplugin/myplugin_test/regression/list.txt)
/// Each line of the "list.txt" file must contain: a local path to the scene, the number of simulation steps to run, and a numerical epsilon for comparison.
/// e.g. "gravity.scn 5 1e-10" to run the scene "regression/gravity.scn" for 5 time steps, and the state difference must be smaller than 1e-10
///
/// As an example, have a look to SofaTest_test/regression
///
/// @author Matthieu Nesme
/// @date 2015
class Regression_test: public BaseSimulationTest, public ::testing::WithParamInterface<RegressionTestData>
{
public:
    static std::string getTestName(const testing::TestParamInfo<RegressionTestData>& p)
    {
        std::string path = p.param.fileScenePath;
        std::cout << "getTestName: " << path << std::endl;

        std::size_t pos = path.find_last_of("/");      // position of "live" in str
        std::string str3 = path.substr(pos+1);
        str3 = str3.substr(0, str3.find_last_of("."));

        return str3;
    }


    void runRegressionScene(RegressionTestData data)
    {
        msg_info("Regression_test") << "  Testing " << data.fileScenePath;

        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();

        simulation::Simulation* simulation = simulation::getSimulation();

        // Load the scene
        sofa::simulation::Node::SPtr root = simulation->load(data.fileScenePath.c_str());

        simulation->init(root.get());

        // TODO lancer visiteur pour dumper MO
        // comparer ce dump avec le fichier sceneName.regressionreference

        bool initializing = false;

        if (helper::system::FileSystem::exists(data.fileRefPath) && !helper::system::FileSystem::isDirectory(data.fileRefPath))
        {
            // Add CompareState components: as it derives from the ReadState, we use the ReadStateActivator to enable them.
            sofa::component::misc::CompareStateCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
//            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(data.fileRefPath);
            compareVisitor.execute(root.get());

            sofa::component::misc::ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
            v_read.execute(root.get());
        }
        else // create reference
        {
            msg_warning("Regression_test") << "Non existing reference created: " << data.fileRefPath;

            // just to create an empty file to know it is already init
            std::ofstream filestream(data.fileRefPath.c_str());
            filestream.close();

            initializing = true;
            sofa::component::misc::WriteStateCreator writeVisitor(sofa::core::ExecParams::defaultInstance());
//            writeVisitor.setCreateInMapping(true);
            writeVisitor.setSceneName(data.fileRefPath);
            writeVisitor.execute(root.get());

            sofa::component::misc::WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
            v_write.execute(root.get());
        }

        for( unsigned int i=0 ; i<data.steps ; ++i )
        {
            simulation->animate( root.get(), root->getDt() );
        }

        if( !initializing )
        {
            // Read the final error: the summation of all the error made at each time step
            sofa::component::misc::CompareStateResult result(sofa::core::ExecParams::defaultInstance());
            result.execute(root.get());

            double errorByDof = result.getErrorByDof() / double(result.getNumCompareState());
            if( errorByDof > data.epsilon )
            {
                msg_error("Regression_test") << data.fileScenePath << ":" << msgendl
                    << "    TOTALERROR: " << result.getTotalError() << msgendl
                    << "    ERRORBYDOF: " << errorByDof;
           }
        }

        // Clear and prepare for next scene
        simulation->unload(root.get());
        root.reset();
    }


};


struct Regression_test_lists
{
    std::vector<RegressionTestData> listScenes;
protected:
    void runRegressionList(const std::string& testDir)
    {
        // lire plugin_test/regression_scene_list -> (file,nb time steps,epsilon)
        // pour toutes les scenes

        const std::string regression_scene_list = testDir + "/list.txt";

        if (helper::system::FileSystem::exists(regression_scene_list) && !helper::system::FileSystem::isDirectory(regression_scene_list))
        {
            msg_info("Regression_test") << "Parsing " << regression_scene_list;

            // parser le fichier -> (file,nb time steps,epsilon)
            std::ifstream iniFileStream(regression_scene_list.c_str());
            while (!iniFileStream.eof())
            {
                std::string line;
                std::string scene;
                unsigned int steps;
                double epsilon;

                getline(iniFileStream, line);
                std::istringstream lineStream(line);
                lineStream >> scene;
                lineStream >> steps;
                lineStream >> epsilon;

                scene = testDir + "/" + scene;
                std::string reference = testDir + "/" + getFileName(scene) + ".reference";

#ifdef WIN32
                // Minimize absolute scene path to avoid MAX_PATH problem
                if (scene.length() > MAX_PATH)
                {
                    ADD_FAILURE() << scene << ": path is longer than " << MAX_PATH;
                    continue;
                }
                char buffer[MAX_PATH];
                GetFullPathNameA(scene.c_str(), MAX_PATH, buffer, nullptr);
                scene = std::string(buffer);
                std::replace(scene.begin(), scene.end(), '\\', '/');
#endif // WIN32
                listScenes.push_back(RegressionTestData(scene, reference, steps, epsilon));
                //runRegressionScene( reference, scene, steps, epsilon );
            }
        }
    }



    void runRegressionTests(const std::string& directory)
    {
        // pour tous plugins/projets
        std::vector<std::string> dirContents;
        helper::system::FileSystem::listDirectory(directory, dirContents);

        for (const std::string& dirContent : dirContents)
        {
            if (helper::system::FileSystem::isDirectory(directory + "/" + dirContent))
            {
                const std::string testDir = directory + "/" + dirContent + "/" + dirContent + "_test/regression";
                if (helper::system::FileSystem::exists(testDir) && helper::system::FileSystem::isDirectory(testDir))
                {
                    runRegressionList(testDir);
                }
            }
        }
    }


    // Create the context for the scene
    virtual void testAll()
    {
        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();

        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        static const std::string regressionsDir = std::string(SOFA_SRC_DIR) + "/applications";
        if (helper::system::FileSystem::exists(regressionsDir))
            runRegressionTests(regressionsDir);

        static const std::string pluginsDir = std::string(SOFA_SRC_DIR) + "/applications/plugins";
        if (helper::system::FileSystem::exists(pluginsDir))
            runRegressionTests(pluginsDir);

        static const std::string devPluginsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/plugins";
        if (helper::system::FileSystem::exists(devPluginsDir))
            runRegressionTests(devPluginsDir);

        static const std::string projectsDir = std::string(SOFA_SRC_DIR) + "/applications/projects";
        if (helper::system::FileSystem::exists(projectsDir))
            runRegressionTests(projectsDir);

        static const std::string devProjectsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/projects";
        if (helper::system::FileSystem::exists(devProjectsDir))
            runRegressionTests(devProjectsDir);

        static const std::string modulesDir = std::string(SOFA_SRC_DIR) + "/modules";
        if (helper::system::FileSystem::exists(modulesDir))
            runRegressionTests(modulesDir);

        static const std::string kernelModulesDir = std::string(SOFA_SRC_DIR) + "/SofaKernel/modules";
        if (helper::system::FileSystem::exists(kernelModulesDir))
            runRegressionTests(kernelModulesDir);
    }

    std::string getFileName(const std::string& s)
    {
        char sep = '/';

        size_t i = s.rfind(sep, s.length());
        if (i != std::string::npos)
        {
            return(s.substr(i + 1, s.length() - i));
        }

        return s;
    }
};

static struct Regression_Sofa_tests : public Regression_test_lists
{
    Regression_Sofa_tests()
    {
        testAll();
    }
} regression_tests;




} // namespace sofa

#endif
