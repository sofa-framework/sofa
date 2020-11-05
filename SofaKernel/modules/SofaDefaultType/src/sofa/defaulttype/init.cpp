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
#include "init.h"

#include <sofa/helper/init.h>
#include <iostream>
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
namespace sofa
{

namespace defaulttype
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

SOFA_DEFAULTTYPE_API void init()
{
    if (!s_initialized)
    {
        sofa::helper::init();
        s_initialized = true;

        std::cout << "Module: None" << std::endl;
        for(auto& info : sofa::defaulttype::DataTypeInfoRegistry::GetRegisteredTypes())
        {
            std::string s = (info->ValidInfo())?"-":"!";
            std::cout << "  " << s << " "  << info->name() << std::endl;
        }

        std::cout << "Module: " << sofa_tostring(SOFA_TARGET) << std::endl;
        for(auto& info : sofa::defaulttype::DataTypeInfoRegistry::GetRegisteredTypes(sofa_tostring(SOFA_TARGET)))
        {
            std::string s = (info->ValidInfo())?"-":"!";
            std::cout << "  " << s << " "  << info->name() << std::endl;
        }
    }
}

SOFA_DEFAULTTYPE_API bool isInitialized()
{
    return s_initialized;
}

SOFA_DEFAULTTYPE_API void cleanup()
{
    if (!s_cleanedUp)
    {
        sofa::helper::cleanup();
        s_cleanedUp = true;
    }
}

SOFA_DEFAULTTYPE_API bool isCleanedUp()
{
    return s_cleanedUp;
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck() {}
    ~CleanupCheck()
    {
        if (defaulttype::isInitialized() && !defaulttype::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaDefaultType", "sofa::defaulttype::cleanup()");
    }
} check;

} // namespace defaulttype

} // namespace sofa
