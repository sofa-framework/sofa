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

#include <sofa/helper/fwd.h>
#include <sofa/helper/AdvancedTimer.h>
namespace sofa::helper::advancedtimer
{


void begin(const char* idStr)
{
    AdvancedTimer::begin(idStr);
}

void end(const char* idStr)
{
    AdvancedTimer::end(idStr);
}

void step(const char* idStr)
{
    AdvancedTimer::step(idStr);
}

void valSet(const char* idStr, double val)
{
    AdvancedTimer::valSet(idStr, val);
}

void valAdd(const char* idStr, double val)
{
    AdvancedTimer::valAdd(idStr, val);
}

void setEnabled(const char* id, bool val)
{
    AdvancedTimer::setEnabled(id, val);
}

void stepBegin(const char* idStr)
{
    AdvancedTimer::stepBegin(idStr);
}

void stepEnd(const char* idStr)
{
    AdvancedTimer::stepEnd(idStr);
}

void stepNext(const char* prevIdStr, const char* nextIdStr)
{
    AdvancedTimer::stepNext(prevIdStr, nextIdStr);
}

void stepBegin(const std::string& idStr)
{
    AdvancedTimer::stepBegin(idStr);
}

void stepEnd(const std::string& idStr)
{
    AdvancedTimer::stepEnd(idStr);
}

void stepNext(const std::string& prevIdStr, const std::string& nextIdStr)
{
    AdvancedTimer::stepNext(prevIdStr, nextIdStr);
}

void stepBegin(const std::string& idStr,const std::string& extra)
{
    AdvancedTimer::stepBegin(idStr, extra);
}

void stepEnd(const std::string& idStr, const std::string& extra)
{
    AdvancedTimer::stepEnd(idStr, extra);
}

}

