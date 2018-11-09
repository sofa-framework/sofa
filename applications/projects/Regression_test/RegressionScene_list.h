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
#ifndef SOFA_RegressionScene_list_H
#define SOFA_RegressionScene_list_H

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/testing/BaseTest.h>

#include <fstream>
#include <SofaTest/Sofa_test.h>

namespace sofa 
{

/// a struct to store all info to perform the regression test
struct RegressionSceneTest_Data
{
    RegressionSceneTest_Data(const std::string& fileScenePath, const std::string& fileRefPath, unsigned int steps, double epsilon)
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

/**
    This class will parse a given list of path (project env paths) and search for list of scene files to collect
    for future regression test.
    Main method is @sa collectScenesFromPaths
    All collected data will be store inside the vector @sa m_listScenes
*/
class RegressionScene_list
{
public:
    /// name of the file list 
    std::string m_listFilename;

    /// List of regression Data to perform @sa RegressionSceneTest_Data
    std::vector<RegressionSceneTest_Data> m_listScenes;

protected:
    /// Method called by collectScenesFromDir to search specific regression file list inside a directory
    void collectScenesFromList(const std::string& testDir)
    {
        // lire plugin_test/regression_scene_list -> (file,nb time steps,epsilon)
        // pour toutes les scenes

        const std::string regression_scene_list = testDir + "/" + m_listFilename;

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

                scene = std::string(SOFA_SRC_DIR) + "/" + scene;
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
                m_listScenes.push_back(RegressionSceneTest_Data(scene, reference, steps, epsilon));
            }
        }
    }


    /// Method called by @sa collectScenesFromPaths to loop on the subdirectories to find regression file list
    void collectScenesFromDir(const std::string& directory)
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
                    collectScenesFromList(testDir);
                }
            }
        }
    }


    /// Main method to start the parsing of regression file list on specific Sofa src paths
    virtual void collectScenesFromPaths(const std::string& listFilename)
    {
        m_listFilename = listFilename;

        static const std::string regressionsDir = std::string(SOFA_SRC_DIR) + "/applications";
        if (helper::system::FileSystem::exists(regressionsDir))
            collectScenesFromDir(regressionsDir);

        static const std::string pluginsDir = std::string(SOFA_SRC_DIR) + "/applications/plugins";
        if (helper::system::FileSystem::exists(pluginsDir))
            collectScenesFromDir(pluginsDir);

        static const std::string devPluginsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/plugins";
        if (helper::system::FileSystem::exists(devPluginsDir))
            collectScenesFromDir(devPluginsDir);

        static const std::string projectsDir = std::string(SOFA_SRC_DIR) + "/applications/projects";
        if (helper::system::FileSystem::exists(projectsDir))
            collectScenesFromDir(projectsDir);

        static const std::string devProjectsDir = std::string(SOFA_SRC_DIR) + "/applications-dev/projects";
        if (helper::system::FileSystem::exists(devProjectsDir))
            collectScenesFromDir(devProjectsDir);

        static const std::string modulesDir = std::string(SOFA_SRC_DIR) + "/modules";
        if (helper::system::FileSystem::exists(modulesDir))
            collectScenesFromDir(modulesDir);

        static const std::string kernelModulesDir = std::string(SOFA_SRC_DIR) + "/SofaKernel/modules";
        if (helper::system::FileSystem::exists(kernelModulesDir))
            collectScenesFromDir(kernelModulesDir);
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



} // namespace sofa

#endif // SOFA_RegressionScene_list_H
