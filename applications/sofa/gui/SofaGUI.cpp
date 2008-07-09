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
#include "SofaGUI.h"
#include <string.h>
namespace sofa
{

namespace gui
{

const char* SofaGUI::programName = NULL;
std::string SofaGUI::guiName = "";
std::vector<std::string> SofaGUI::guiOptions;
SofaGUI* SofaGUI::currentGUI = NULL;

//std::list<SofaGUI::GUICreator> SofaGUI::guiCreators;
std::list<SofaGUI::GUICreator>& SofaGUI::guiCreators()
{
    static std::list<SofaGUI::GUICreator> creators;
    return creators;
}

void SofaGUI::SetProgramName(const char* argv0)
{
    if (argv0)
        programName = argv0;
}

const char* SofaGUI::GetProgramName()
{
    return programName;
}

std::vector<std::string> SofaGUI::ListSupportedGUI()
{
    std::vector<std::string> names;
    for(std::list<GUICreator>::iterator it = guiCreators().begin(), itend = guiCreators().end(); it != itend; ++it)
    {
        names.push_back(it->name);
    }
    return names;
}

std::string SofaGUI::ListSupportedGUI(char separator)
{
    std::string names;
    bool first = true;
    for(std::list<GUICreator>::iterator it = guiCreators().begin(), itend = guiCreators().end(); it != itend; ++it)
    {
        if (!first) names += separator; else first = false;
        names += it->name;
    }
    return names;
}

const char* SofaGUI::GetGUIName()
{
    const char* name = guiName.c_str();
    if (!name[0] && !guiCreators().empty())
    {
        std::list<GUICreator>::iterator it = guiCreators().begin();
        std::list<GUICreator>::iterator itend = guiCreators().end();
        name = it->name;
        int prio = it->priority;
        while (++it != itend)
        {
            if (it->priority > prio)
            {
                name = it->name;
                prio = it->priority;
            }
        }
    }
    return name;
}

void SofaGUI::SetGUIName(const char* name)
{
    guiName = name;
}

void SofaGUI::AddGUIOption(const char* option)
{
    guiOptions.push_back(option);
}

SofaGUI::GUICreator* SofaGUI::GetGUICreator(const char* name)
{
    if (!name) name = GetGUIName();
    std::list<GUICreator>::iterator it = guiCreators().begin();
    std::list<GUICreator>::iterator itend = guiCreators().end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it == itend)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" not found."<<std::endl;
        std::cerr << "Available GUIs:" << ListSupportedGUI(' ') << std::endl;
        return NULL;
    }
    else
        return &(*it);
}

SofaGUI* SofaGUI::CurrentGUI()
{
    return currentGUI;
}

void SofaGUI::Redraw()
{
    if (currentGUI) currentGUI->redraw();
}

sofa::simulation::Node* SofaGUI::CurrentSimulation()
{
    if (currentGUI)
        return currentGUI->currentSimulation();
    else
        return NULL;
}

int SofaGUI::RegisterGUI(const char* name, CreateGUIFn* creator, InitGUIFn* init, int priority)
{
    std::list<GUICreator>::iterator it = guiCreators().begin();
    std::list<GUICreator>::iterator itend = guiCreators().end();
    while (it != itend && strcmp(name, it->name))
        ++it;
    if (it != itend)
    {
        std::cerr << "ERROR(SofaGUI): GUI "<<name<<" duplicate registration."<<std::endl;
        return 1;
    }
    GUICreator entry;
    entry.name = name;
    entry.creator = creator;
    entry.init = init;
    entry.priority = priority;
    guiCreators().push_back(entry);
    return 0;
}

SofaGUI::SofaGUI()
{
    if (currentGUI)
        std::cerr << "WARNING(SofaGUI): multiple concurrent active gui." << std::endl;
    else
        currentGUI = this;
}

SofaGUI::~SofaGUI()
{
    if (currentGUI == this)
        currentGUI = NULL;
}

} // namespace gui

} // namespace sofa
