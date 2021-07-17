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
#pragma once

#include <sofa/helper/config.h>
#include <string>
namespace sofa::helper
{
class StateMask;
class ColorMap;
class MarchingCubeUtility;

namespace advancedtimer
{
SOFA_HELPER_API void begin(const char* idStr);
SOFA_HELPER_API void end(const char* idStr);
SOFA_HELPER_API void step(const char* idStr);

SOFA_HELPER_API void valSet(const char* idStr, double val);
SOFA_HELPER_API void valAdd(const char* idStr, double val);

SOFA_HELPER_API void setEnabled(const char *id, bool val);

SOFA_HELPER_API void stepBegin(const char* idStr);
SOFA_HELPER_API void stepEnd(const char* idStr);
SOFA_HELPER_API void stepNext(const char* idStr, const char*idStrp);

SOFA_HELPER_API void stepBegin(const std::string& idStr,const std::string& extra);
SOFA_HELPER_API void stepEnd(const std::string& idStr, const std::string& extra);

SOFA_HELPER_API void stepBegin(const std::string& idStr);
SOFA_HELPER_API void stepEnd(const std::string& idStr);
SOFA_HELPER_API void stepNext(const std::string& idStr, const std::string& idStrp);
}
}

namespace sofa::helper::visual
{
class DrawTool;
}
