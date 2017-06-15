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
#include <SofaGui/config.h>
#include "Main.h"
#include "GUIManager.h"

#include "BatchGUI.h"
#ifdef SOFA_GUI_QT
#include "qt/RealGUI.h"
#endif
#ifdef SOFA_GUI_GLUT
#include "glut/SimpleGUI.h"
#endif

#ifdef SOFA_GUI_GLUT
#ifdef SOFA_HAVE_BOOST
#include "glut/MultithreadGUI.h"
#endif
#endif

namespace sofa
{

namespace gui
{

void initMain()
{
    // This function does nothing. It is used to make sure that this file is linked, so that the following GUI registrations are made.
    static bool first = true;
    if (first)
    {

        first = false;
    }
}

int BatchGUIClass = GUIManager::RegisterGUI("batch", &BatchGUI::CreateGUI, &BatchGUI::InitGUI, -1);

#ifdef SOFA_GUI_GLUT
int SimpleGUIClass = GUIManager::RegisterGUI("glut", &glut::SimpleGUI::CreateGUI, &glut::SimpleGUI::InitGUI, 0);

#ifdef SOFA_HAVE_BOOST
int MtGUIClass = GUIManager::RegisterGUI("glut-mt", &glut::MultithreadGUI::CreateGUI, &glut::MultithreadGUI::InitGUI, 0);
#endif
#endif

#ifdef SOFA_GUI_QGLVIEWER
int QGLViewerGUIClass = GUIManager::RegisterGUI ( "qglviewer", &qt::RealGUI::CreateGUI, &qt::RealGUI::InitGUI, 3 );
#endif

#ifdef SOFA_GUI_QTVIEWER
int QtGUIClass = GUIManager::RegisterGUI ( "qt", &qt::RealGUI::CreateGUI, &qt::RealGUI::InitGUI, 2 );
#endif

} // namespace gui

} // namespace sofa
