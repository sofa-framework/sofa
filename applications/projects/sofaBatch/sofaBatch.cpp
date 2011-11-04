/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/component/misc/WriteState.h>

#include <ctime>

using std::cerr; using std::endl;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------


void apply(std::string &input, unsigned int nbsteps, std::string &output)
{
    cerr<<"\n****SIMULATION*  (.scn:"<< input<<", #steps:"<<nbsteps<<", .simu:"<<output<<")"<<endl;

    sofa::simulation::xml::numDefault = 0;

    using namespace sofa::helper::system;
    sofa::simulation::tree::GNode* groot = NULL;

    groot = dynamic_cast< sofa::simulation::tree::GNode* >( sofa::simulation::tree::getSimulation()->load(input.c_str()));
    sofa::simulation::tree::getSimulation()->init(groot);
    if (groot == NULL)
    {
        std::cerr << "CANNOT open " << input << " !\n";
        return;
    }



    std::string outputdir = SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str()) + std::string("/applications/projects/sofaBatch/simulation/");


    std::string mstate = outputdir + SetDirectory::GetFileName(output.c_str());


    sofa::component::misc::WriteStateCreator visitor;
    visitor.setSceneName(mstate);
    visitor.execute(groot);


    sofa::component::misc::WriteStateActivator v_write(true);
    v_write.execute(groot);


    clock_t curtime = clock();
    std::cout << "Computing " <<  nbsteps << " for " << input <<  std::endl;
    for (unsigned int j=0; j<nbsteps; j++)
    {
        sofa::simulation::tree::getSimulation()->animate(groot);
    }
    double t = (clock() - curtime)/((double)CLOCKS_PER_SEC);

    std::cout << nbsteps << " steps done in " << t  << " seconds (average="<<t/nbsteps<<" s/step)" <<std::endl;




    std::string simulationFileName = mstate + std::string(".simu") ;


    std::ofstream out(simulationFileName.c_str());
    if (!out.fail())
    {
        out << input.c_str() << " Init: 0.000 s End: " << nbsteps*sofa::simulation::tree::getSimulation()->getContext()->getDt() << " s " << sofa::simulation::tree::getSimulation()->getContext()->getDt() << " baseName: "<<mstate;
        out.close();

        std::cout << "Simulation parameters saved in "<<simulationFileName<<std::endl;
    }
    else
    {
        std::cout<<simulationFileName<<" file error\n";
    }



    sofa::simulation::tree::getSimulation()->unload(groot);


}

int main(int argc, char** argv)
{
//     sofa::helper::BackTrace::autodump();

    std::vector<std::string> files;
    std::string fileName ;


    sofa::helper::parse(&files, "\nThis is a SOFA batch that permits to run and to save simulation states without GUI.\nGive a name file containing actions == list of (input .scn, #simulated time steps, output .simu). See file tasks for an example.\n\nHere are the command line arguments")
    (argc,argv);


    if (!files.empty())
        fileName = files[0];
    else
    {
        cerr<<"No input tasks file\nsee help\n";
        return 0;
    }


    sofa::helper::system::DataRepository.findFile(fileName);


    //Get the list of scenes to test
    std::ifstream end(fileName.c_str());
    std::string input;
    int nbsteps;
    std::string output;

    while( end >> input && end >>nbsteps && end >> output  )
    {
        sofa::helper::system::DataRepository.findFile(input);
        apply(input, nbsteps, output);
    }
    end.close();





    return 0;
}
