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
#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/WriteState.h>
#include <sofa/component/misc/CompareState.h>
#include <sofa/component/misc/ReadTopology.h>
#include <sofa/component/misc/WriteTopology.h>
#include <sofa/component/misc/CompareTopology.h>

#include <ctime>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

void apply(std::vector< std::string> &files, unsigned int iterations, bool reinit, bool useTopology)
{
    std::cout <<"******* Args ********" << std::endl;
    std::cout <<"files: " << files[0] << std::endl;
    std::cout <<"iterations: " << iterations << std::endl;
    std::cout <<"reinit: " << reinit << std::endl;
    std::cout <<"useTopology: " << useTopology << std::endl;
    std::cout <<"*********************" << std::endl;


    using namespace sofa::helper::system;
    sofa::simulation::tree::GNode* groot = NULL;

    //Launch the comparison for each scenes
    for (unsigned int i=0; i<files.size(); ++i)
    {
        groot = dynamic_cast< sofa::simulation::tree::GNode* >( sofa::simulation::getSimulation()->load(files[i].c_str()));
        sofa::simulation::tree::getSimulation()->init(groot);

        if (groot == NULL)
        {
            std::cerr << "CANNOT open " << files[i] << " !\n";
            continue;
        }


        //Filename where the simulation of the current scene will be saved (in Sofa/applications/projects/sofaVerification/simulation/)
        std::string file = SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
        file = file + std::string("/applications/projects/sofaVerification/simulation/") + SetDirectory::GetFileName(files[i].c_str());


        //If we initialize the system, we add only WriteState components, to store the reference states
        if (reinit)
        {
            if (useTopology)
            {
                sofa::component::misc::WriteTopologyCreator writeVisitor;

                writeVisitor.setCreateInMapping(true);
                writeVisitor.setSceneName(file);
                writeVisitor.execute(groot);

                sofa::component::misc::WriteTopologyActivator v_write(true);
                v_write.execute(groot);
            }
            else
            {
                sofa::component::misc::WriteStateCreator writeVisitor;

                writeVisitor.setCreateInMapping(true);
                writeVisitor.setSceneName(file);
                writeVisitor.execute(groot);

                sofa::component::misc::WriteStateActivator v_write(true);
                v_write.execute(groot);
            }

        }
        else
        {
            if (useTopology)
            {
                //We add CompareTopology components: as it derives from the ReadTopology, we use the ReadTopologyActivator to enable them.
                sofa::component::misc::CompareTopologyCreator compareVisitor;
                compareVisitor.setCreateInMapping(true);
                compareVisitor.setSceneName(file);
                compareVisitor.execute(groot);

                sofa::component::misc::ReadTopologyActivator v_read(true);
                v_read.execute(groot);
            }
            else
            {
                //We add CompareState components: as it derives from the ReadState, we use the ReadStateActivator to enable them.
                sofa::component::misc::CompareStateCreator compareVisitor;
                compareVisitor.setCreateInMapping(true);
                compareVisitor.setSceneName(file);
                compareVisitor.execute(groot);

                sofa::component::misc::ReadStateActivator v_read(true);
                v_read.execute(groot);
            }
        }


        //Save the initial time
        clock_t curtime = clock();

        //Do as many iterations as specified in entry of the program. At each step, the compare state will compare the computed states to the recorded states
        std::cout << "Computing " <<  iterations << " for " << files[i] <<  std::endl;
        for (unsigned int i=0; i<iterations; i++) sofa::simulation::getSimulation()->animate(groot);
        double t = (clock() - curtime)/((double)CLOCKS_PER_SEC);

        std::cout <<"ITERATIONS " <<  iterations << " TIME " << t  << " seconds" <<std::endl;


        //We read the final error: the summation of all the error made at each time step
        if (!reinit)
        {
            if (useTopology)
            {
                sofa::component::misc::CompareTopologyResult result;
                result.execute(groot);
                std::cout << "ERROR " << result.getTotalError() << "\n";

                const std::vector<unsigned int> listResult = result.getErrors();
                if (listResult.size() != 5)
                {
                    std::cout << "ERROR while reading detail of errors by topological element." << std::endl;
                    break;
                }

                std::cout << "ERROR by element type " << "\n";
                std::cout << "EDGES " << listResult[0]/(double)result.getNumCompareTopology() << "\n";
                std::cout << "TRIANGLES " << listResult[1]/(double)result.getNumCompareTopology() << "\n";
                std::cout << "QUADS " << listResult[2]/(double)result.getNumCompareTopology() << "\n";
                std::cout << "TETRAHEDRA " << listResult[3]/(double)result.getNumCompareTopology() << "\n";
                std::cout << "HEXAHEDRA " << listResult[4]/(double)result.getNumCompareTopology() << "\n";
            }
            else
            {
                sofa::component::misc::CompareStateResult result;
                result.execute(groot);
                std::cout << "ERROR " << result.getTotalError() << "\n";
                std::cout << "ERRORBYDOF " << result.getErrorByDof()/(double)result.getNumCompareState() << "\n";
            }
        }

        //Clear and prepare for next scene
        sofa::simulation::getSimulation()->unload(groot);
    }
}

int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    std::string fileName ;
    std::vector<std::string> files;
    unsigned int iterations=100;
    bool reinit=false;
    bool topology=false;


    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::helper::parse(&files, "This is a SOFA verification. To use it, specify in the command line a \".ini\" file containing the path to the scenes you want to test. ")
    .option(&reinit,'r',"reinit","Recreate the references state files")
    .option(&iterations, 'i',"iteration", "Number of iterations for testing")
    .option(&topology, 't',"topology", "Specific mode to run tests on topology")
    (argc,argv);


    if (!files.empty()) fileName = files[0];

    files.clear();


    sofa::helper::system::DataRepository.findFile(fileName);
    std::string extension=fileName.substr(fileName.size()-4);
    std::cout << "Extension : " << extension << "\n";
    if (extension == std::string(".ini"))
    {
        //Get the list of scenes to test
        std::ifstream end(fileName.c_str());
        std::string s;
        while( end >> s )
        {
            sofa::helper::system::DataRepository.findFile(s);
            files.push_back(s);
        }
        end.close();
    }
    else
        files.push_back(fileName);


    std::cout << "Files : " << files[0] << "\n";

    apply(files, iterations, reinit, topology);



    return 0;
}
