/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>


#ifndef WIN32
#define SOFA_EXPORT_DYNAMIC_LIBRARY
#define SOFA_IMPORT_DYNAMIC_LIBRARY
#define SOFA_PLUGINEXAMPLE_API
#else
#ifdef SOFA_BUILD_PLUGINEXAMPLE
#define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
#define SOFA_PLUGINEXAMPLE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
#define SOFA_PLUGINEXAMPLE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif
#endif


namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_PLUGINEXAMPLE_API void initExternalModule();
    SOFA_PLUGINEXAMPLE_API const char* getModuleName();
    SOFA_PLUGINEXAMPLE_API const char* getModuleVersion();
    SOFA_PLUGINEXAMPLE_API const char* getModuleLicense();
    SOFA_PLUGINEXAMPLE_API const char* getModuleDescription();
    SOFA_PLUGINEXAMPLE_API const char* getModuleComponentList();
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
    return "PluginExample";
}

const char* getModuleVersion()
{
    return "0.2";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "a simple example of a plugin component module";
}

const char* getModuleComponentList()
{
    return "PendulumMapping, MyFakeComponent, OtherFakeComponent";
}



}

}


SOFA_LINK_CLASS(PendulumMapping)
SOFA_LINK_CLASS(MyFakeComponent)
SOFA_LINK_CLASS(OtherFakeComponent)

