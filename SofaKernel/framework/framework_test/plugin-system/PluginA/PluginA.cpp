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
#include "PluginA.h"
#include <sofa/core/Plugin.h>

#include <PluginMonitor/PluginMonitor.h>

static struct PluginAMonitor {
    PluginAMonitor() { PluginA_loaded++; }
    ~PluginAMonitor() { PluginA_unloaded++; }
} PluginAMonitor_;

class PluginA: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginA);
public:
    PluginA(): Plugin("PluginA") {
        addComponent<Foo>("Component Foo");
    }
};

int FooClass = PluginA::registerObject("Component Foo")
.add<Foo>();

int BarClass = PluginA::registerObject("Component Bar")
.add< Bar<float> >(true)
.add< Bar<double> >();

SOFA_PLUGIN_ENTRY_POINT(PluginA);
