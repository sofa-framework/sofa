/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef REQUIREDPLUGIN_H_
#define REQUIREDPLUGIN_H_
#include "config.h"

#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/BaseObject.h>



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
    sofa::core::objectmodel::Data<helper::vector<std::string>> d_pluginName;
protected:
    RequiredPlugin();
    virtual ~RequiredPlugin() {}

public:

    virtual void parse(sofa::core::objectmodel::BaseObjectDescription* arg);

    static void loadPlugin( const std::string& pluginName );

};

}

}

}

#endif /* REQUIREDPLUGIN_H_ */
