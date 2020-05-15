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
#include <sofa/helper/system/config.h>
#include <SofaComponentAll/initComponentAll.h>

#include <SofaBase/initSofaBase.h>
#include <SofaCommon/initSofaCommon.h>
#include <SofaGeneral/initSofaGeneral.h>
#include <SofaMisc/initSofaMisc.h>

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

/// Convenient functions to help user to know what contains the plugin
extern "C" {
    SOFA_SOFACOMPONENTALL_API void initExternalModule();
    SOFA_SOFACOMPONENTALL_API const char* getModuleName();
    SOFA_SOFACOMPONENTALL_API const char* getModuleVersion();
    SOFA_SOFACOMPONENTALL_API const char* getModuleLicense();
    SOFA_SOFACOMPONENTALL_API const char* getModuleDescription();
    SOFA_SOFACOMPONENTALL_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if(!first) return;
    first = false;

/// TODO: remove SofaAllCommonComponents backward compatibility at SOFA v20.06
#ifdef SOFACOMPONENTALL_USING_DEPRECATED_NAME
    msg_deprecated("SofaAllCommonComponents") << "This plugin was renamed into SofaComponentAll. Backward compatiblity will be stopped at SOFA v20.06";
#endif

    sofa::component::initSofaBase();
    sofa::component::initSofaCommon();
    sofa::component::initSofaGeneral();
    sofa::component::initSofaMisc();
}

const char* getModuleName()
{
    return "SofaComponentAll";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This package exposes all SOFA components.";
}

const char* getModuleComponentList()
{
    return "";
}


} /// namespace component

} /// namespace sofa
