/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_Regression_test_H
#define SOFA_Regression_test_H

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaLoader/ReadState.h>
#include <SofaExporter/WriteState.h>
#include <SofaValidation/CompareState.h>

#include <SofaTest/Sofa_test.h>

namespace sofa {

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
class Regression_test: public testing::Test
{

protected:

    void runRegressionScene( const std::string& testScene, unsigned nbsteps, double epsilon )
    {

        std::cerr<<"Regression_test : testing scene "<<testScene<<" "<<nbsteps<<std::endl;

        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();

        simulation::Simulation* simulation = simulation::getSimulation();

        //Load the scene
        sofa::simulation::Node::SPtr root = simulation->load(testScene.c_str());

        simulation->init(root.get());


        // TODO lancer visiteur pour dumper MO
        // comparer ce dump avec le fichier sceneName.regressionreference

        std::string reference = testScene + ".reference";

        bool initializing = false;

        if (helper::system::FileSystem::exists(reference) && !helper::system::FileSystem::isDirectory(reference))
        {
            //We add CompareState components: as it derives from the ReadState, we use the ReadStateActivator to enable them.
            sofa::component::misc::CompareStateCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
//            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(reference);
            compareVisitor.execute(root.get());

            sofa::component::misc::ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
            v_read.execute(root.get());
        }
        else // create reference
        {
            std::cerr<<"REGRESSION TEST : WARNING a reference is not existing and is created now, \""<<testScene<<".reference*\" files should be added to the repository "<<std::endl;


            // just to create an empty file to know it is already init
            std::ofstream filestream(reference.c_str());
            filestream.close();

            initializing = true;
            sofa::component::misc::WriteStateCreator writeVisitor(sofa::core::ExecParams::defaultInstance());
//            writeVisitor.setCreateInMapping(true);
            writeVisitor.setSceneName(reference);
            writeVisitor.execute(root.get());

            sofa::component::misc::WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance() /* PARAMS FIRST */, true);
            v_write.execute(root.get());
        }

        for( unsigned int i=0 ; i<nbsteps ; ++i )
        {
            simulation->animate( root.get(), root->getDt() );
        }



        if( !initializing )
        {
            //We read the final error: the summation of all the error made at each time step
            sofa::component::misc::CompareStateResult result(sofa::core::ExecParams::defaultInstance());
            result.execute(root.get());

            if( result.getTotalError() > epsilon )
            {
                ADD_FAILURE() << "ERROR "<< testScene;

//                << " " << result.getTotalError() << " ERRORBYDOF "
//                                                             << static_cast<double>(result.getErrorByDof())
//                                                             / result.getNumCompareState() << std::endl;
            }

        }

        //Clear and prepare for next scene
        simulation->unload(root.get());
        root.reset();
    }




    void runRegressionList( const std::string& testDir )
    {
        // lire plugin_test/regression_scene_list -> (file,nb time steps,epsilon)
        // pour toutes les scenes

        const std::string regression_scene_list = testDir + "/list.txt";

        if (helper::system::FileSystem::exists(regression_scene_list) && !helper::system::FileSystem::isDirectory(regression_scene_list))
        {
            std::cerr<<"Regression_test : testing list "<<regression_scene_list<<std::endl;

            // parser le fichier -> (file,nb time steps,epsilon)
            std::ifstream iniFileStream(regression_scene_list.c_str());
            while (!iniFileStream.eof())
            {
                std::string line;
                std::string currentScene;
                unsigned int nbsteps;
                double epsilon;
                getline(iniFileStream, line);
                std::istringstream lineStream(line);
                lineStream >> currentScene;
                lineStream >> nbsteps;
                lineStream >> epsilon;
//                DataRepository.findFile(currentScene);

                runRegressionScene( testDir + "/" + currentScene, nbsteps, epsilon );
            }

        }
    }



    void testTestPath( const std::string& pluginsDirectory )
    {
        // pour tous plugins/projets
        std::vector<std::string> dir;
        helper::system::FileSystem::listDirectory(pluginsDirectory, dir);

        for (std::vector<std::string>::iterator i = dir.begin(); i != dir.end(); ++i)
        {
            const std::string pluginPath = pluginsDirectory + "/" + *i;
            if (helper::system::FileSystem::isDirectory(pluginPath))
            {
//                std::cerr<<"Regression_test : testing plugin/project "<<pluginPath<<std::endl;

                const std::string testDir = pluginPath + "/" + *i + "_test/regression";
                if (helper::system::FileSystem::exists(testDir) && helper::system::FileSystem::isDirectory(testDir))
                {
                    runRegressionList( testDir );
                }
            }
        }
    }


    // Create the context for the scene
    virtual void SetUp()
    {
        sofa::component::initComponentBase();
        sofa::component::initComponentCommon();
        sofa::component::initComponentGeneral();
        sofa::component::initComponentAdvanced();
        sofa::component::initComponentMisc();

        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        // pour tous les emplacements critiques
        static const std::string pluginsDir = std::string(SOFA_SRC_DIR) + "/applications/plugins";
        if (helper::system::FileSystem::exists(pluginsDir))
            testTestPath(pluginsDir);

        static const std::string devPluginsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/plugins";
        if (helper::system::FileSystem::exists(devPluginsDir))
            testTestPath(devPluginsDir);

        static const std::string projectsDir = std::string(SOFA_SRC_DIR) + "/applications/projects";
        if (helper::system::FileSystem::exists(projectsDir))
            testTestPath(projectsDir);

        static const std::string devProjectsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/projects";
        if (helper::system::FileSystem::exists(devProjectsDir))
            testTestPath(devProjectsDir);

    }

};




} // namespace sofa

#endif
