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
#include "PluginMonitor.h"

int SOFA_PLUGINMONITOR_API PluginA_loaded;
int SOFA_PLUGINMONITOR_API PluginA_unloaded;
int SOFA_PLUGINMONITOR_API PluginB_loaded;
int SOFA_PLUGINMONITOR_API PluginB_unloaded;
int SOFA_PLUGINMONITOR_API PluginC_loaded;
int SOFA_PLUGINMONITOR_API PluginC_unloaded;
int SOFA_PLUGINMONITOR_API PluginD_loaded;
int SOFA_PLUGINMONITOR_API PluginD_unloaded;
int SOFA_PLUGINMONITOR_API PluginE_loaded;
int SOFA_PLUGINMONITOR_API PluginE_unloaded;
int SOFA_PLUGINMONITOR_API PluginF_loaded;
int SOFA_PLUGINMONITOR_API PluginF_unloaded;

void reset_plugin_monitor() {
    PluginA_loaded = 0;
    PluginA_unloaded = 0;
    PluginB_loaded = 0;
    PluginB_unloaded = 0;
    PluginC_loaded = 0;
    PluginC_unloaded = 0;
    PluginD_loaded = 0;
    PluginD_unloaded = 0;
    PluginE_loaded = 0;
    PluginE_unloaded = 0;
    PluginF_loaded = 0;
    PluginF_unloaded = 0;
}
