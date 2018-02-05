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
#include <sofa/core/Plugin.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include <PluginMonitor/PluginMonitor.h>
#include <PluginA/PluginA.h>


static struct PluginDMonitor {
    PluginDMonitor() { PluginD_loaded++; }
    ~PluginDMonitor() { PluginD_unloaded++; }
} PluginDMonitor_;


class FooD: public Foo {
};

template <class C>
class BazD: public sofa::core::objectmodel::BaseObject {
};

class VecD {
};


class PluginD: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginD);
public:
    PluginD(): Plugin("PluginD") {
    }
};

int FooDClass = PluginD::registerObject("Component FooD")
  .add<FooD>();

int BarClass = PluginD::registerObject("")
.add< Bar<VecD> >();

int BazDClass = PluginD::registerObject("Component BazD")
.add< BazD<float> >(true)
.add< BazD<double> >();

SOFA_PLUGIN_ENTRY_POINT(PluginD);
