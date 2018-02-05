/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_SENSABLEPLUGIN_API
#else
#ifdef SOFA_BUILD_SENSABLEPLUGIN
#define SOFA_SENSABLEPLUGIN_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_SENSABLEPLUGIN_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif

#ifdef WIN32
// BUGFIX(Jeremie A. 02-05-2009): put OpenHaptics libs here instead of the project file to work around a bug in qmake when there is a space in an environment variable used to look-up a library
#pragma comment(lib,"hl.lib")
#pragma comment(lib,"hd.lib")
#pragma comment(lib,"hdu.lib")
#endif


namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_SENSABLEPLUGIN_API void initExternalModule();
    SOFA_SENSABLEPLUGIN_API const char* getModuleName();
    SOFA_SENSABLEPLUGIN_API const char* getModuleVersion();
    SOFA_SENSABLEPLUGIN_API const char* getModuleLicense();
    SOFA_SENSABLEPLUGIN_API const char* getModuleDescription();
    SOFA_SENSABLEPLUGIN_API const char* getModuleComponentList();
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
    return "Plugin Sensable";
}

const char* getModuleVersion()
{
    return "beta 1.1";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "Force feedback with sensable devices into SOFA Framework";
}

const char* getModuleComponentList()
{
    return "ForceFeedback controllers ";
}

}

}


SOFA_LINK_CLASS(NewOmniDriver)
SOFA_LINK_CLASS(OmniDriver)
