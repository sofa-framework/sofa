/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/WriteState.h>
#include <sofa/component/misc/CompareState.h>
#include <sofa/component/misc/ReadTopology.h>
#include <sofa/component/misc/WriteTopology.h>
#include <sofa/component/misc/CompareTopology.h>
#include <sofa/component/init.h>
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

void apply(const std::string& directory, std::vector<std::string>& files,
        unsigned int iterations, bool reinit, bool useTopology)
{

    sofa::component::init(); // ensures all components are initialized, also introduce a dependency to all libraries, avoiding problems with -Wl,--as-needed flag

    sofa::simulation::Simulation* simulation = sofa::simulation::getSimulation();

    //Launch the comparison for each scenes
    for (unsigned int i = 0; i < files.size(); ++i)
    {

        const std::string& currentFile = files[i];
        GNode* groot = dynamic_cast<GNode*> (simulation->load(currentFile.c_str()));
        if (groot == NULL)
        {
            std::cerr << "CANNOT open " << currentFile << " !" << std::endl;
            continue;
        }

        simulation->init(groot);

        //Filename where the simulation of the current scene will be saved (in Sofa/applications/projects/sofaVerification/simulation/)
        std::string refFile;
        if(directory.empty())
        {
            refFile += SetDirectory::GetParentDir(DataRepository.getFirstPath().c_str());
            refFile += "/applications/projects/sofaVerification/simulation/";
            refFile += SetDirectory::GetFileName(currentFile.c_str());
        }
        else
        {
            refFile += directory;
            refFile += '/';
            refFile += SetDirectory::GetFileName(currentFile.c_str());
        }

        //If we initialize the system, we add only WriteState components, to store the reference states
        if (reinit)
        {
            if (useTopology)
            {
                sofa::component::misc::WriteTopologyCreator writeVisitor(sofa::core::ExecParams::defaultInstance());

                writeVisitor.setCreateInMapping(true);
                writeVisitor.setSceneName(refFile);
                writeVisitor.execute(groot);

                sofa::component::misc::WriteTopologyActivator v_write(true,sofa::core::ExecParams::defaultInstance());
                v_write.execute(groot);
            }
            else
            {
                sofa::component::misc::WriteStateCreator writeVisitor(sofa::core::ExecParams::defaultInstance());

                writeVisitor.setCreateInMapping(true);
                writeVisitor.setSceneName(refFile);
                writeVisitor.execute(groot);

                sofa::component::misc::WriteStateActivator v_write(true,sofa::core::ExecParams::defaultInstance());
                v_write.execute(groot);
            }

        }
        else
        {
            if (useTopology)
            {
                //We add CompareTopology components: as it derives from the ReadTopology, we use the ReadTopologyActivator to enable them.
                sofa::component::misc::CompareTopologyCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
                compareVisitor.setCreateInMapping(true);
                compareVisitor.setSceneName(refFile);
                compareVisitor.execute(groot);

                sofa::component::misc::ReadTopologyActivator v_read(sofa::core::ExecParams::defaultInstance(),true);
                v_read.execute(groot);
            }
            else
            {
                //We add CompareState components: as it derives from the ReadState, we use the ReadStateActivator to enable them.
                sofa::component::misc::CompareStateCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
                compareVisitor.setCreateInMapping(true);
                compareVisitor.setSceneName(refFile);
                compareVisitor.execute(groot);

                sofa::component::misc::ReadStateActivator v_read(true,sofa::core::ExecParams::defaultInstance());
                v_read.execute(groot);
            }
        }

        //Do as many iterations as specified in entry of the program. At each step, the compare state will compare the computed states to the recorded states
        std::cout << "Computing " << iterations << " for " << currentFile << std::endl;

        //Save the initial time
        clock_t curtime = clock();
        for (unsigned int i = 0; i < iterations; i++)
        {
            simulation->animate(groot);
        }
        double t = static_cast<double>(clock() - curtime) / CLOCKS_PER_SEC;

        std::cout << "ITERATIONS " << iterations << " TIME " << t << " seconds ";

        //We read the final error: the summation of all the error made at each time step
        if (!reinit)
        {
            if (useTopology)
            {
                sofa::component::misc::CompareTopologyResult result(sofa::core::ExecParams::defaultInstance());
                result.execute(groot);
                std::cout << "ERROR " << result.getTotalError() << ' ';

                const std::vector<unsigned int>& listResult = result.getErrors();
                if (listResult.size() != 5)
                {
                    std::cerr
                            << "ERROR while reading detail of errors by topological element."
                                    << std::endl;
                    break;
                }

                std::cout
                        << "ERROR by element type "
                                << " EDGES "      << static_cast<double>(listResult[0])
                                / result.getNumCompareTopology()
                                << " TRIANGLES "  << static_cast<double>(listResult[1])
                                / result.getNumCompareTopology()
                                << " QUADS "      << static_cast<double>(listResult[2])
                                / result.getNumCompareTopology()
                                << " TETRAHEDRA " << static_cast<double>(listResult[3])
                                / result.getNumCompareTopology()
                                << " HEXAHEDRA "  << static_cast<double>(listResult[4])
                                / result.getNumCompareTopology();
            }
            else
            {
                sofa::component::misc::CompareStateResult result(sofa::core::ExecParams::defaultInstance());
                result.execute(groot);
                std::cout
                        << "ERROR " << result.getTotalError() << " ERRORBYDOF "
                                << static_cast<double>(result.getErrorByDof())
                                / result.getNumCompareState() << std::endl;
            }
        }
        std::cout << std::endl;

        //Clear and prepare for next scene
        simulation->unload(groot);
        delete groot;
    }
}

int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    std::string directory;
    std::vector<std::string> fileArguments;
    std::vector<std::string> sceneFiles;
    std::vector<std::string> plugins;
    unsigned int iterations = 100;
    bool reinit = false;
    bool topology = false;
    unsigned lifetime = 0;

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::helper::parse(
        &fileArguments,
        "This is SOFA verification. "
        "To use it, specify in the command line the scene files you want to test, "
        "or a \".ini\" file containing the path to the scenes.")
    .option(&reinit, 'r', "reinit", "Recreate the references state files")
    .option(&iterations, 'i', "iteration", "Number of iterations for testing")
    .option(&directory, 'd', "directory", "The directory for reference files")
    .option(&plugins,'p',"plugin","load given plugins")
    .option(&topology, 't', "topology", "Specific mode to run tests on topology")
    .option(&lifetime, 'l', "lifetime", "Maximum execution time in seconds (default: 0 -> no limit")
    (argc, argv);

#ifdef SOFA_HAVE_BOOST
    sofa::helper::system::thread::TimeoutWatchdog watchdog;
    if(lifetime > 0)
    {
        watchdog.start(lifetime);
    }
#endif

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());


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
                    << "iterations: "  << iterations << '\n'
                    << "reinit: "      << reinit     << '\n'
                    << "useTopology: " << topology   << '\n'
                    << "files : "                    << '\n';
    for(size_t i = 0; i < sceneFiles.size(); ++i)
    {
        std::cout << "  " << sceneFiles[i] << '\n';
    }

    apply(directory, sceneFiles, iterations, reinit, topology);

    return 0;
}
