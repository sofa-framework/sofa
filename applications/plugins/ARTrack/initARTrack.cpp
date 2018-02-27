/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/config.h>


#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_ARTRACKPLUGIN_API
#else
#ifdef SOFA_BUILD_ARTRACKPLUGIN
#define SOFA_ARTRACKPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_ARTRACKPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_ARTRACKPLUGIN_API void initExternalModule();
    SOFA_ARTRACKPLUGIN_API const char* getModuleName();
    SOFA_ARTRACKPLUGIN_API const char* getModuleVersion();
	SOFA_ARTRACKPLUGIN_API const char* getModuleLicense();
    SOFA_ARTRACKPLUGIN_API const char* getModuleDescription();
    SOFA_ARTRACKPLUGIN_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "Plugin ARTrack";
}

const char* getModuleVersion()
{
    return "beta 1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "ARTtrack port into SOFA Framework";
}

const char* getModuleComponentList()
{
    return "ARTrackDriver";
}



}

}


SOFA_LINK_CLASS(ARTrackDriver)
SOFA_LINK_CLASS(ARTrackController)
