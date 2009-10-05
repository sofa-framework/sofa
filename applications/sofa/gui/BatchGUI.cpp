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
#include "BatchGUI.h"
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace gui
{

BatchGUI::BatchGUI()
    : groot(NULL), nbIter(500)
{
}

BatchGUI::~BatchGUI()
{
}

int BatchGUI::mainLoop()
{
    if (groot)
    {
        sofa::simulation::Node::ctime_t tSpent = sofa::helper::system::thread::CTime::getRefTime();
        std::cout << "Computing "<<nbIter<<" iterations." << std::endl;
        for (int i=0; i<nbIter; i++)
        {
            sofa::simulation::getSimulation()->animate(groot);
        }
        std::cout << nbIter << " iterations done in "<< 1000.0*(sofa::helper::system::thread::CTime::getRefTime()-tSpent)/((double)sofa::helper::system::thread::CTime::getTicksPerSec()) << std::endl;
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

void BatchGUI::setScene(sofa::simulation::Node* groot, const char* filename, bool )
{
    this->groot = groot;
    this->filename = (filename?filename:"");
}

sofa::simulation::Node* BatchGUI::currentSimulation()
{
    return groot;
}


int BatchGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& /*options*/)
{
    return 0;
}

SofaGUI* BatchGUI::CreateGUI(const char* name, const std::vector<std::string>& /*options*/, sofa::simulation::Node* groot, const char* filename)
{
    BatchGUI::guiName = name;
    BatchGUI* gui = new BatchGUI();
    gui->setScene(groot, filename);
    return gui;
}

} // namespace gui

} // namespace sofa
