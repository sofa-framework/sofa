/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "BatchGUI.h"
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#ifdef SOFA_SMP
#include <athapascan-1>
#endif
#include <sofa/helper/system/thread/CTime.h>
#include <iostream>
#include <sstream>

namespace sofa
{

namespace gui
{

const signed int BatchGUI::DEFAULT_NUMBER_OF_ITERATIONS = 1000;
signed int BatchGUI::nbIter = BatchGUI::DEFAULT_NUMBER_OF_ITERATIONS;

BatchGUI::BatchGUI()
    : groot(NULL)
{
}

BatchGUI::~BatchGUI()
{
}

int BatchGUI::mainLoop()
{
    if (groot)
    {   
        if (nbIter >= 0)
        {
            sofa::simulation::getSimulation()->animate(groot.get());
            //As no visualization is done by the Batch GUI, these two lines are not necessary.
            sofa::simulation::getSimulation()->updateVisual(groot.get());
            std::cout << "Computing "<<nbIter<<" iterations."<< std::endl;
            sofa::simulation::Visitor::ctime_t rtfreq = sofa::helper::system::thread::CTime::getRefTicksPerSec();
            sofa::simulation::Visitor::ctime_t tfreq = sofa::helper::system::thread::CTime::getTicksPerSec();
            sofa::simulation::Visitor::ctime_t rt = sofa::helper::system::thread::CTime::getRefTime();
            sofa::simulation::Visitor::ctime_t t = sofa::helper::system::thread::CTime::getFastTime();
                for (signed int i=0; i < nbIter; i++)
                {
                    sofa::simulation::getSimulation()->animate(groot.get());
                    //As no visualization is done by the Batch GUI, these two lines are not necessary.
                    sofa::simulation::getSimulation()->updateVisual(groot.get());
                }

            t = sofa::helper::system::thread::CTime::getFastTime()-t;
            rt = sofa::helper::system::thread::CTime::getRefTime()-rt;
            std::cout << nbIter << " iterations done in "<< ((double)t)/((double)tfreq) << " s ( " << (((double)tfreq)*nbIter)/((double)t) << " FPS)." << std::endl;
            std::cout << nbIter << " iterations done in "<< ((double)rt)/((double)rtfreq) << " s ( " << (((double)rtfreq)*nbIter)/((double)rt) << " FPS)." << std::endl;
        }

        else if (nbIter == -1)
        {   
            unsigned int nbIterInf = 0;
            unsigned int i = 0;

            sofa::simulation::getSimulation()->animate(groot.get());
            //As no visualization is done by the Batch GUI, these two lines are not necessary.
            sofa::simulation::getSimulation()->updateVisual(groot.get());
            std::cout << "Computing infinite iterations."<< std::endl;

            sofa::simulation::Visitor::ctime_t rtfreq = sofa::helper::system::thread::CTime::getRefTicksPerSec();
            sofa::simulation::Visitor::ctime_t tfreq = sofa::helper::system::thread::CTime::getTicksPerSec();
            sofa::simulation::Visitor::ctime_t rt = sofa::helper::system::thread::CTime::getRefTime();
            sofa::simulation::Visitor::ctime_t t = sofa::helper::system::thread::CTime::getFastTime();
                do
                {
                    sofa::simulation::getSimulation()->animate(groot.get());
                    //As no visualization is done by the Batch GUI, these two lines are not necessary.
                    sofa::simulation::getSimulation()->updateVisual(groot.get());
                    
                    if (i%1000 == 0 && i != 0)
                    {   
                        nbIterInf = i;
                        t = sofa::helper::system::thread::CTime::getFastTime()-t;
                        rt = sofa::helper::system::thread::CTime::getRefTime()-rt;
                        std::cout << nbIterInf << " iterations done in "<< ((double)t)/((double)tfreq) << " s ( " << (((double)tfreq)*nbIterInf)/((double)t) << " FPS)." << std::endl;
                        std::cout << nbIterInf << " iterations done in "<< ((double)rt)/((double)rtfreq) << " s ( " << (((double)rtfreq)*nbIterInf)/((double)rt) << " FPS)." << std::endl;
                    }
                    //After every 1000 iterations, time taken to process will be shown.
                    i++;
                }while(1);
            
        }   
    }
    return 0;
}

void BatchGUI::redraw()
{
}

int BatchGUI::closeGUI()
{
    delete this;
    return 0;
}

void BatchGUI::setScene(sofa::simulation::Node::SPtr groot, const char* filename, bool )
{
    this->groot = groot;
    this->filename = (filename?filename:"");

    resetScene();
}


void BatchGUI::resetScene()
{
    sofa::simulation::Node* root = currentSimulation();

    if ( root )
    {
        root->setTime(0.);
        simulation::getSimulation()->reset ( root );

        sofa::simulation::UpdateSimulationContextVisitor(sofa::core::ExecParams::defaultInstance()).execute(root);
    }
}

void BatchGUI::startDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Node* root = currentSimulation();
    if (root)
    {
        m_dumpVisitorStream.str("");
        sofa::simulation::Visitor::startDumpVisitor(&m_dumpVisitorStream, root->getTime());
    }
#endif
}

void BatchGUI::stopDumpVisitor()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    sofa::simulation::Visitor::stopDumpVisitor();
    m_dumpVisitorStream.flush();
    m_dumpVisitorStream.str("");
#endif
}

sofa::simulation::Node* BatchGUI::currentSimulation()
{
    return groot.get();
}


int BatchGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& options)
{   
    setNumIterations(DEFAULT_NUMBER_OF_ITERATIONS);
    
    //parse options
    for (unsigned int i=0 ; i<options.size() ; i++)
    {
        size_t cursor = 0;
        std::string opt = options[i];
        //Set number of iterations
        //(option = "nbIterations=N where N is the number of iterations)
        if ( (cursor = opt.find("nbIterations=")) != std::string::npos )
        {
            signed int nbIterations;
            std::istringstream iss;
            iss.str(opt.substr(cursor+std::string("nbIterations=").length(), std::string::npos));
            iss >> nbIterations;
            
            if (nbIterations == -1)
            {
                setNumIterations(-1);
            }
            //When iterations (n = -1) is set to -1, infinite iterations will take place
            else
                 setNumIterations(nbIterations);
        }
    }
    return 0;
}

BaseGUI* BatchGUI::CreateGUI(const char* name, const std::vector<std::string>& /*options*/, sofa::simulation::Node::SPtr groot, const char* filename)
{
    BatchGUI::mGuiName = name;
    BatchGUI* gui = new BatchGUI();
    gui->setScene(groot, filename);
    return gui;
}

} // namespace gui

} // namespace sofa
