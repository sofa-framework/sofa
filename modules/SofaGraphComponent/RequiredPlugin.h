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
#ifndef REQUIREDPLUGIN_H_
#define REQUIREDPLUGIN_H_
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/BaseVector.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_GRAPH_COMPONENT_API RequiredPlugin : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(RequiredPlugin,core::objectmodel::BaseObject);
    Data<helper::vector<std::string> > d_pluginName; ///< plugin name (or several names if you need to load different plugins or a plugin with several alternate names)
    Data<std::string> d_searchPath; ///< Directory to scan before the default ones

    /// standard->custom suffixes pairs (to be used if the plugin is compiled outside of Sofa
    /// with a non standard way of differenciating versions), using ! to represent empty suffix
    Data<helper::vector<helper::fixed_array<std::string,2> > > d_suffixMap;

    Data<bool> d_stopAfterFirstNameFound; ///< Stop after the first plugin name that is loaded successfully
    Data<bool> d_stopAfterFirstSuffixFound; ///< For each plugin name, stop after the first suffix that is loaded successfully
    Data<bool> d_requireOne; ///< Display an error message if no plugin names were successfully loaded
    Data<bool> d_requireAll; ///< Display an error message if any plugin names failed to be loaded

protected:
    RequiredPlugin();
    virtual ~RequiredPlugin() {}

public:

    virtual void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override;

    void loadPlugin();

};

}

}

}

#endif /* REQUIREDPLUGIN_H_ */
