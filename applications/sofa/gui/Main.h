/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GUI_MAIN_H
#define SOFA_GUI_MAIN_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_GUIMAIN
#	define SOFA_GUIMAIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_GUIMAIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif


namespace sofa
{

namespace gui
{

/// necessary to register all the available GUI
void SOFA_GUIMAIN_API initMain();

}
}
#endif
