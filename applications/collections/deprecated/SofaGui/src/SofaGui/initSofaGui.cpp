/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaGui/initSofaGui.h>

#include <sofa/gui/common/GUIManager.h>

#include <sofa/gui/batch/init.h>
#if SOFAGUI_HAVE_SOFA_GUI_QT
#include <sofa/gui/qt/init.h>
#endif
#if SOFAGUI_HAVE_SOFA_GUI_HEADLESSRECORDER
#include <sofa/gui/headlessrecorder/init.h>
#endif

namespace sofa::gui
{

void initSofaGui()
{
    static bool first = true;
    if (first)
    {
        sofa::gui::batch::init();
#if SOFAGUI_HAVE_SOFA_GUI_QT
        sofa::gui::qt::init();
#endif
#if SOFAGUI_HAVE_SOFA_GUI_HEADLESSRECORDER
        sofa::gui::headlessrecorder::init();
#endif
        first = false;
    }
}

} // namespace sofa::gui
