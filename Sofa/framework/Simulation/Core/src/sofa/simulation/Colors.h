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

#include <sofa/simulation/config.h>
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

// Register a new color, without a name
SOFA_SIMULATION_CORE_API sofa::Index registerColor(const std::string& hexColor);

// Register a new color, if the name already exists, the old color is overriden
SOFA_SIMULATION_CORE_API sofa::Index registerColor(const std::string& classname, const std::string& hexColor);

// Returns the color associated with the given classname. Throw std::runtime_exception otherwise
SOFA_SIMULATION_CORE_API const char* getColor(const std::string& classname);

// Returns the color associated with the given COLORID. Throw std::runtime_exception otherwise
SOFA_SIMULATION_CORE_API const char* getColor(const sofa::Index userID);

// Returns wether or not there is a color associated with this name
SOFA_SIMULATION_CORE_API bool hasColor(const std::string& className);

// Returns wether or not there is a color associated with this name
SOFA_SIMULATION_CORE_API bool hasColor(const sofa::Index& id);

// This is to allow old code to still work,
class SOFA_SIMULATION_CORE_API DeprecatedColor
{
public:
    const char* operator[](size_t);
};
SOFA_SIMULATION_CORE_API extern DeprecatedColor COLOR;


}
