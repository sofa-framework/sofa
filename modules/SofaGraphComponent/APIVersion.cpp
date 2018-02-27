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
/******************************************************************************
*  Contributors:                                                              *
*  - damien.marchal@univ-lille1.fr                                            *
******************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseContext ;

#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;

#include "APIVersion.h"

namespace sofa
{

namespace component
{

namespace _apiversion_
{

APIVersion::APIVersion() :
     d_level ( initData(&d_level, std::string("17.06"), "level", "The API Level of the scene ('17.06', '17.12', '18.06', ...)"))
{
}

APIVersion::~APIVersion()
{
}

void APIVersion::init()
{
    Inherit1::init();
    checkInputData() ;
}

void APIVersion::checkInputData()
{
    if(!d_level.isSet() && !name.isSet() ){
        msg_warning() << "The level is not set. Using 17.06 as default value. " ;
        return ;
    }
    if( !d_level.isSet() && name.isSet() ){
        d_level.setValue(getName());
    }
    std::vector<std::string> allowedVersion = { "17.06", "17.12", "18.06", "18.12" } ;
    if( std::find( allowedVersion.begin(), allowedVersion.end(), d_level.getValue()) == allowedVersion.end() )
    {
        msg_warning() << "The provided level '"<< d_level.getValue() <<"' is now valid. " ;
    }
}

const std::string& APIVersion::getApiLevel()
{
    return d_level.getValue() ;
}

SOFA_DECL_CLASS(APIVersion)
int APIVersionClass = core::RegisterObject("Specify the APIVersion of the component used in a scene.")
        .add< APIVersion >();

} // namespace _apiversion_

} // namespace component

} // namespace sofa
