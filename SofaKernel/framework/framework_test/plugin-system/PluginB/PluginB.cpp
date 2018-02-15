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
#include "PluginB.h"
#include <PluginMonitor/PluginMonitor.h>

static struct PluginBMonitor {
    PluginBMonitor() { PluginB_loaded++; }
    ~PluginBMonitor() { PluginB_unloaded++; }
} PluginBMonitor_;

extern "C" {

void initExternalModule()
{
}

const char* getModuleName()
{
    return "PluginB";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "None";
}

const char* getModuleDescription()
{
    return "This is a empty, old-style, plugin";
}

const char* getModuleComponentList()
{
    return "";
}

}
