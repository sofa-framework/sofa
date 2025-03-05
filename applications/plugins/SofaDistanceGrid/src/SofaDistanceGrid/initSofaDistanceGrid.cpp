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
#include <SofaDistanceGrid/initSofaDistanceGrid.h>
#include "components/collision/DistanceGridCollisionModel.h"
#include "components/forcefield/DistanceGridForceField.h"
#include "RegisterModelToCollisionFactory.h"

namespace sofadistancegrid
{
extern "C" {
SOFA_SOFADISTANCEGRID_API void initExternalModule();
SOFA_SOFADISTANCEGRID_API const char* getModuleName();
SOFA_SOFADISTANCEGRID_API const char* getModuleVersion();
SOFA_SOFADISTANCEGRID_API const char* getModuleLicense();
SOFA_SOFADISTANCEGRID_API const char* getModuleDescription();
SOFA_SOFADISTANCEGRID_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaDistanceGrid();
}

void initSofaDistanceGrid()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
    sofa::component::collision::registerDistanceGridCollisionModel();
}

const char* getModuleName()
{
    return sofadistancegrid::MODULE_NAME;
}

const char* getModuleVersion()
{
    return sofadistancegrid::MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "A distance grid stores the distance to an object into a 3d regular grid.  "
           "This is an efficient data structure to get a distance approximation for   "
           "point in space. This is why it is often used to implement collisions.     ";
}

} /// namespace sofadistancegrid
