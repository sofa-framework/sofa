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

#include <cstring>
#include <vector>
#include <string>

namespace sofa::simulation::Colors
{

enum COLORID
{
    NODE,
    OBJECT,
    CONTEXT,
    BMODEL,
    CMODEL,
    MMODEL,
    PROJECTIVECONSTRAINTSET,
    CONSTRAINTSET,
    IFFIELD,
    FFIELD,
    SOLVER,
    COLLISION,
    MMAPPING,
    MAPPING,
    MASS,
    TOPOLOGY,
    VMODEL,
    LOADER,
    CONFIGURATIONSETTING,
    ALLCOLORS
};

size_t registerColor(const std::string& hexColor);
void registerColor(const std::string& classname, const std::string& hexColor);
const char* getColor(const std::string& classname);
const char* getColor(const COLORID classType);

// This is to allow old code to still work,
class DeprecatedColor
{
public:
    const char* operator[](size_t);
};
extern DeprecatedColor COLOR;


}
