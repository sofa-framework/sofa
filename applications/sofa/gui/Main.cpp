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
#include <SofaGui/config.h>
#include "Main.h"
#include "GUIManager.h"

#include "BatchGUI.h"
#if SOFAGUI_HAVE_SOFAGUIQT
#include "qt/RealGUI.h"
#endif
#if SOFAGUI_HAVE_SOFAHEADLESSRECORDER
#include "headlessRecorder/HeadlessRecorder.h"
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

int BatchGUIClass = GUIManager::RegisterGUI("batch", &BatchGUI::CreateGUI, &BatchGUI::RegisterGUIParameters, -1);

#if SOFAGUI_HAVE_SOFAHEADLESSRECORDER
int HeadlessRecorderClass = GUIManager::RegisterGUI ( "hRecorder", &hRecorder::HeadlessRecorder::CreateGUI, &hRecorder::HeadlessRecorder::RegisterGUIParameters, 2 );
#endif
  
#if SOFAGUIQT_HAVE_QGLVIEWER
int QGLViewerGUIClass = GUIManager::RegisterGUI ( "qglviewer", &qt::RealGUI::CreateGUI, nullptr, 3 );
#endif

#if SOFAGUIQT_HAVE_QTVIEWER
int QtGUIClass = GUIManager::RegisterGUI ( "qt", &qt::RealGUI::CreateGUI, nullptr, 2 );
#endif

} // namespace gui

} // namespace sofa
