/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/common/GUIManager.h>


static sofa::gui::common::BaseGUI* CreateFakeGUI(const char* /*name*/, sofa::simulation::Node::SPtr /*groot*/, const char* /*filename*/)
{
    return new FakeGUI();
}

void FakeGUI::Create()
{
    // sofa FakeGUI
    sofa::gui::common::GUIManager::RegisterGUI("fake", CreateFakeGUI, NULL);
    sofa::gui::common::GUIManager::Init(nullptr,"fake");
    sofa::gui::common::GUIManager::createGUI(NULL,NULL);
}

void FakeGUI::sendMessage(const std::string & msgType,const std::string & msgValue)
{
    printf("FakeGUI::sendMessage(\"%s\",\"%s\")\n",msgType.c_str(),msgValue.c_str());
}
