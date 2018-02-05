/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaSimulationTree/GNode.h>
#include <SofaSimulationTree/init.h>
#include <SofaLoader/ReadState.h>
#include <SofaExporter/WriteState.h>
#include <SofaValidation/CompareState.h>
#include <SofaLoader/ReadTopology.h>
#include <SofaExporter/WriteTopology.h>
#include <SofaValidation/CompareTopology.h>
#include <sofa/helper/system/thread/TimeoutWatchdog.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <iostream>
#include <fstream>
#include <ctime>

using sofa::helper::system::DataRepository;
using sofa::helper::system::SetDirectory;
using sofa::simulation::tree::GNode;



#ifndef WIN32
#include <dlfcn.h>
bool loadPlugin(const char* filename)
{
    void *handle;
    handle=dlopen(filename, RTLD_LAZY);
    if (!handle)
    {
        std::cerr<<"Error loading plugin "<<filename<<": "<<dlerror()<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#else
bool loadPlugin(const char* filename)
{
    HINSTANCE DLLHandle;
    DLLHandle = LoadLibraryA(filename); //warning: issue between unicode and ansi encoding on Visual c++ -> force to ansi-> dirty!
    if (DLLHandle == NULL)
    {
        std::cerr<<"Error loading plugin "<<filename<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#endif


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

void apply(const std::string& /*directory*/, std::vector<std::string>& files, bool /*reinit*/)
{


    sofa::simulation::Simulation* simulation = sofa::simulation::getSimulation();

    //Launch the comparison for each scenes
    for (unsigned int i = 0; i < files.size(); ++i)
    {

        const std::string& currentFile = files[i];
        GNode::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<GNode> (simulation->load(currentFile.c_str()));
        if (groot == NULL)
        {
            std::cerr << "CANNOT open " << currentFile << " !" << std::endl;
            continue;
        }
        std::cout << "sec. Loading " << currentFile << std::endl;

        //Save the initial time
        clock_t curtime = clock();
        simulation->init(groot.get());
        double t = static_cast<double>(clock() - curtime) / (float)CLOCKS_PER_SEC;
        std::cout << "Init time " << t << " sec." << std::endl;

        //Clear and prepare for next scene
        simulation->unload(groot);
        groot.reset();
    }
}

int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::helper::BackTrace::autodump();

    std::string refdir;
    std::string dataPath;
    std::vector<std::string> fileArguments;
    std::vector<std::string> sceneFiles;
    std::vector<std::string> plugins;
    bool reinit = false;
    unsigned lifetime = 0;

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::helper::parse(
        &fileArguments,
        "This is SOFA verification. "
        "To use it, specify in the command line the scene files you want to test, "
        "or a \".ini\" file containing the path to the scenes.")
    .option(&reinit,     'r', "reinit",    "Recreate the references state files")
    .option(&refdir,     'd', "refdir",    "The directory for reference files")
    .option(&dataPath,   'a', "datapath",  "A colon-separated (semi-colon on Windows) list of directories to search for data files (scenes, resources...)")
    .option(&plugins,    'p', "plugin",    "Load given plugins")
    .option(&lifetime,   'l', "lifetime",  "Maximum execution time in seconds (default: 0 -> no limit")
    (argc, argv);

#ifdef SOFA_HAVE_BOOST
    sofa::helper::system::thread::TimeoutWatchdog watchdog;
    if(lifetime > 0)
    {
        watchdog.start(lifetime);
    }
#endif

    for(unsigned int i = 0; i < plugins.size(); i++)
    {
        loadPlugin(plugins[i].c_str());
    }

    DataRepository.addLastPath(dataPath);
    for(size_t i = 0; i < fileArguments.size(); ++i)
    {
        std::string currentFile = fileArguments[i];
        DataRepository.findFile(currentFile);

        if (currentFile.compare(currentFile.size() - 4, 4, ".ini") == 0)
        {
            //This is an ini file: get the list of scenes to test
            std::ifstream iniFileStream(currentFile.c_str());
            while (!iniFileStream.eof())
            {
                std::string line;
                std::string currentScene;
                // extracting the filename line by line because each line can contain
                // extra data, ignored by this program but that may be useful for
                // other tools.
                getline(iniFileStream, line);
                std::istringstream lineStream(line);
                lineStream >> currentScene;
                DataRepository.findFile(currentScene);
                sceneFiles.push_back(currentScene);
            }
        }
        else
        {
            // this is supposed to be a scene file
            sceneFiles.push_back(currentFile);
        }
    }

    std::cout
            << "*********************************************************************\n"
                    << "******* Arguments ***************************************************\n"
                    << "reinit: "      << reinit     << '\n'
                    << "files : "                    << '\n';
    for(size_t i = 0; i < sceneFiles.size(); ++i)
    {
        std::cout << "  " << sceneFiles[i] << '\n';
    }

    apply(refdir, sceneFiles, reinit);

    sofa::simulation::tree::cleanup();
    return 0;
}
