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
#include <sofa/gui/init.h>

#include <sofa/gui/component/init.h>
#include <sofa/gui/common/init.h>
#include <sofa/gui/batch/init.h>
#include <sofa/gui/qt/init.h>
#if SOFA_GUI_HAVE_SOFA_GUI_HEADLESSRECORDER
#include <sofa/gui/headlessrecorder/init.h>
#endif

namespace sofa::gui
{

void init()
{
    static bool first = true;
    if (first)
    {
        sofa::gui::component::init();
        sofa::gui::common::init();
        sofa::gui::batch::init();
        sofa::gui::qt::init();
#if SOFA_GUI_HAVE_SOFA_GUI_HEADLESSRECORDER
        sofa::gui::headlessrecorder::init();
#endif
        first = false;
    }
}

} // namespace sofa::gui
