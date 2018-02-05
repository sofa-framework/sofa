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
#include <iostream>
#include <fstream>
#include <ctime>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/PluginManager.h>

#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>



#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <SofaExporter/WriteState.h>



using std::cerr;
using std::endl;
using std::cout;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------


void apply(std::string &input, unsigned int nbsteps, std::string &output)
{
    cout<<"\n****SIMULATION*  (.scn:"<< input<<", #steps:"<<nbsteps<<", .simu:"<<output<<")"<<endl;

    // --- Create simulation graph ---
    sofa::simulation::Node::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(input.c_str()));
    if (groot==NULL)
    {
        groot = sofa::simulation::getSimulation()->createNewGraph("");
    }

    if (!groot)
    {
        cerr << "Error, unable to access groot" << std::endl;
        return;
    }

    sofa::simulation::getSimulation()->init(groot.get());
    groot->setAnimate(true);


    // --- Init output file ---
    std::string outputdir =  sofa::helper::system::SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string("/applications/projects/sofaBatch/simulation/");
    std::string mstate = outputdir + sofa::helper::system::SetDirectory::GetFileName(output.c_str());

    // --- Init Write state visitor ---
    sofa::component::misc::WriteStateCreator visitor(sofa::core::ExecParams::defaultInstance());
    visitor.setSceneName(mstate);
    visitor.execute(groot.get());

    sofa::component::misc::WriteStateActivator v_write(sofa::core::ExecParams::defaultInstance(), true);
    v_write.execute(groot.get());


    // --- Sofa GUI Batch animationLoop ---
    sofa::simulation::getSimulation()->animate(groot.get());

    std::cout << "Computing "<<nbsteps<<" iterations." << std::endl;
    sofa::simulation::Visitor::ctime_t rtfreq = sofa::helper::system::thread::CTime::getRefTicksPerSec();
    sofa::simulation::Visitor::ctime_t tfreq = sofa::helper::system::thread::CTime::getTicksPerSec();
    sofa::simulation::Visitor::ctime_t rt = sofa::helper::system::thread::CTime::getRefTime();
    sofa::simulation::Visitor::ctime_t t = sofa::helper::system::thread::CTime::getFastTime();
    for (unsigned int i=0; i<nbsteps; i++)
        sofa::simulation::getSimulation()->animate(groot.get());

    t = sofa::helper::system::thread::CTime::getFastTime()-t;
    rt = sofa::helper::system::thread::CTime::getRefTime()-rt;

    std::cout << nbsteps << " iterations done in "<< ((double)t)/((double)tfreq) << " s ( " << (((double)tfreq)*nbsteps)/((double)t) << " FPS)." << std::endl;
    std::cout << nbsteps << " iterations done in "<< ((double)rt)/((double)rtfreq) << " s ( " << (((double)rtfreq)*nbsteps)/((double)rt) << " FPS)." << std::endl;

    // --- Exporting output simulation ---
    std::string simulationFileName = mstate + std::string(".simu") ;
    std::ofstream out(simulationFileName.c_str());
    if (!out.fail())
    {
        out << input.c_str() << " Init: 0.000 s End: " << nbsteps*groot.get()->getDt() << " s " << groot.get()->getDt() << " baseName: "<<mstate;
        out.close();

        std::cout << "Simulation parameters saved in "<<simulationFileName<<std::endl;
    }
    else
    {
        std::cout<<simulationFileName<<" file error\n";
    }


    sofa::simulation::getSimulation()->unload(groot);

    return;
}


int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    // --- Parameter initialisation ---
    std::vector<std::string> files;
    std::string fileName ;
    std::vector<std::string> plugins;
    std::vector<unsigned int> nbstepsations;

    sofa::helper::parse(&files, "\nThis is a SOFA batch that permits to run and to save simulation states without GUI.\nGive a name file containing actions == list of (input .scn, #simulated time steps, output .simu). See file tasks for an example.\n\nHere are the command line arguments")
    .option(&plugins,'l',"load","load given plugins")
    (argc,argv);


    // --- check input file
    if (!files.empty())
        fileName = files[0];
    else
    {
        cerr<<"No input tasks file\nsee help\n";
        return 0;
    }

    fileName = sofa::helper::system::DataRepository.getFile(fileName);
    //sofa::helper::system::DataRepository.findFile(fileName);


    // --- Init component ---
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());


    // --- plugins ---
    for (unsigned int i=0; i<plugins.size(); i++)
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);

    sofa::helper::system::PluginManager::getInstance().init();


    // --- Perform task list ---
    //std::ifstream end(fileName.c_str());

    std::string input;
//    int nbsteps;
    std::string output;

//    while( end >> input && end >>nbsteps && end >> output  )
//    {
//        std::cout<<"input "<<input<<std::endl;
//        sofa::helper::system::DataRepository.findFile(input);
//        apply(input, nbsteps, output);
//    }
//    end.close();

    std::string strfilename(argv[1]);
    std::string stroutput(argv[3]);
    sofa::helper::system::DataRepository.findFile(strfilename);
    apply(strfilename, atoi(argv[2]), stroutput);

    sofa::simulation::tree::cleanup();
    return 0;
}
