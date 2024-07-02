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
#include <FailingPlugin/FailingPlugin.h>

extern "C" {

static int counter = 0;

SOFA_FAILINGPLUGIN_API void initExternalModule()
{
    static bool first = true;

    if (first)
    {
        first = false;
    }
    counter++;
}

SOFA_FAILINGPLUGIN_API bool moduleIsInitialized()
{
    //always return false for testing
    return false;
}

SOFA_FAILINGPLUGIN_API const char* getModuleName()
{
    return "FailingPlugin";
}

SOFA_FAILINGPLUGIN_API const char* getModuleVersion()
{
    return "0.7";
}

SOFA_FAILINGPLUGIN_API const char* getModuleLicense()
{
    return "LicenseTest";
}

SOFA_FAILINGPLUGIN_API const char* getModuleDescription()
{
    return "A plugin that always fail to initialize";
}

SOFA_FAILINGPLUGIN_API const char* getModuleComponentList()
{
    return "ComponentFailingPlugin";
}

} // extern "C"

namespace failingplugin
{
    SOFA_FAILINGPLUGIN_API void init() {}
}
