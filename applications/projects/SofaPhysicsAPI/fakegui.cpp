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
#include "fakegui.h"
#include <sofa/gui/BaseGUI.h>
#include <sofa/gui/GUIManager.h>


static sofa::gui::BaseGUI* CreateFakeGUI(const char* /*name*/, sofa::simulation::Node::SPtr /*groot*/, const char* /*filename*/)
{
    return new FakeGUI();
}

void FakeGUI::Create()
{
    // sofa FakeGUI
    sofa::gui::GUIManager::RegisterGUI("fake", CreateFakeGUI, NULL);
    sofa::gui::GUIManager::Init(0,"fake");
    sofa::gui::GUIManager::createGUI(NULL,NULL);
}

void FakeGUI::sendMessage(const std::string & msgType,const std::string & msgValue)
{
    printf("FakeGUI::sendMessage(\"%s\",\"%s\")\n",msgType.c_str(),msgValue.c_str());
}
