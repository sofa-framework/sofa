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
#pragma once
#include <sofa/simulation/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/Data.h>

namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API RequiredPlugin : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(RequiredPlugin,core::objectmodel::BaseObject);
    sofa::core::objectmodel::Data<type::vector<std::string> > d_pluginName; ///< plugin name (or several names if you need to load different plugins or a plugin with several alternate names)
    sofa::core::objectmodel::Data<type::vector<type::fixed_array<std::string,2> > > d_suffixMap; ///< standard->custom suffixes pairs (to be used if the plugin is compiled outside of Sofa with a non standard way of differentiating versions), using ! to represent empty suffix

    sofa::core::objectmodel::Data<bool> d_stopAfterFirstNameFound; ///< Stop after the first plugin name that is loaded successfully
    sofa::core::objectmodel::Data<bool> d_stopAfterFirstSuffixFound; ///< For each plugin name, stop after the first suffix that is loaded successfully
    sofa::core::objectmodel::Data<bool> d_requireOne; ///< Display an error message if no plugin names were successfully loaded
    sofa::core::objectmodel::Data<bool> d_requireAll; ///< Display an error message if any plugin names failed to be loaded

    sofa::core::objectmodel::Data<type::vector<std::string> > d_loadedPlugins; ///< List of the plugins that are have been loaded.

protected:
    RequiredPlugin();
    ~RequiredPlugin() override = default;

public:

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override;

    /// load a list of plugins requested in Data
    bool loadPlugin();

};

} // namespace sofa::simulation
