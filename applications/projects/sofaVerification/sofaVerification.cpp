/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/WriteState.h>
#include <sofa/component/misc/CompareState.h>

#include <ctime>

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

void apply(std::vector< std::string> &files, unsigned int iterations, bool reinit)
{
    using namespace sofa::helper::system;
    sofa::simulation::tree::GNode* groot = NULL;
    for (unsigned int i=0; i<files.size(); ++i)
    {
        groot = dynamic_cast< sofa::simulation::tree::GNode* >( sofa::simulation::tree::getSimulation()->load(files[i].c_str()));
        if (groot == NULL)
        {
            std::cerr << "CANNOT open " << files[i] << " !\n";
            continue;
        }



        std::string file = SetDirectory::GetParentDir(sofa::helper::system::DataRepository.getFirstPath().c_str());
        file = file + std::string("/applications/projects/sofaVerification/simulation/") + SetDirectory::GetFileName(files[i].c_str());

        if (reinit)
        {
            sofa::component::misc::WriteStateCreator compareVisitor;
            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(file);
            compareVisitor.execute(groot);

            sofa::component::misc::WriteStateActivator v_write(true);
            v_write.execute(groot);
        }
        else
        {
            sofa::component::misc::CompareStateCreator compareVisitor;
            compareVisitor.setCreateInMapping(true);
            compareVisitor.setSceneName(file);
            compareVisitor.execute(groot);

            sofa::component::misc::ReadStateActivator v_read(true);
            v_read.execute(groot);
        }
        clock_t curtime = clock();
        std::cout << "Computing " <<  iterations << " for " << files[i] <<  std::endl;
        for (unsigned int i=0; i<iterations; i++)
        {
            sofa::simulation::tree::getSimulation()->animate(groot);
        }
        double t = (clock() - curtime)/((double)CLOCKS_PER_SEC);

        std::cout << iterations << " iterations done in " << t  << " seconds" <<std::endl;

        if (!reinit)
        {
            sofa::component::misc::CompareStateResult result;
            result.execute(groot);
            std::cout << "ERROR : " << result.getError() << "\n";

        }
        sofa::simulation::tree::getSimulation()->unload(groot);
    }
}

int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    std::string fileName ;
    std::vector<std::string> files;
    unsigned int iterations=100;
    bool reinit=false;




    sofa::helper::parse(&files, "This is a SOFA verification. Here are the command line arguments")
    .option(&reinit,'r',"reinit","Recreate the references state files")
    .option(&iterations, 'i',"iteration", "Number of iterations for testing")
    (argc,argv);


    if (!files.empty()) fileName = files[0];

    sofa::helper::system::DataRepository.findFile(fileName);


    //Get the list of scenes to test
    files.clear();
    std::ifstream end(fileName.c_str());
    std::string s;
    while( end >> s )
    {
        sofa::helper::system::DataRepository.findFile(s);
        files.push_back(s);
    }
    end.close();


    if (reinit) apply(files, iterations, true);
    else
        apply(files, iterations, false);


    return 0;
}
