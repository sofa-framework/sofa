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

#include <sofa/component/sceneutility/config.h>
#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;
using sofa::core::RegisterObject ;

#include <sofa/component/sceneutility/APIVersion.h>
#include <sofa/version.h>
#include <numeric>

namespace sofa::component::sceneutility::_apiversion_
{

APIVersion::APIVersion() :
     d_level ( initData(&d_level, std::string(SOFA_VERSION_STR), "level", "The API Level of the scene ('17.06', '17.12', '18.06', ...)"))
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
        msg_warning() << "The level is not set. Using " << SOFA_VERSION_STR << " as default value. " ;
        return ;
    }
    if( !d_level.isSet() && name.isSet() ){
        d_level.setValue(getName());
    }

    const auto & API_version = d_level.getValue();
    static const std::set<std::string> allowedAPIVersions { "17.06", "17.12", "18.06", "18.12", "19.06", "19.12", "20.06", "20.12", SOFA_VERSION_STR } ;
    if( allowedAPIVersions.find(API_version) == std::cend(allowedAPIVersions) )
    {
        const auto allowedVersionStr = std::accumulate(std::next(allowedAPIVersions.begin()), allowedAPIVersions.end(), *(allowedAPIVersions.begin()), [](const std::string & s, const std::string & v) {
            return s + ", " + v;
        });
        msg_warning() << "The provided level '"<< API_version <<"' is not valid. Allowed versions are [" << allowedVersionStr << "]." ;
    }
}

const std::string& APIVersion::getApiLevel()
{
    return d_level.getValue() ;
}

int APIVersionClass = core::RegisterObject("Specify the APIVersion of the component used in a scene.")
        .add< APIVersion >();

} // namespace sofa::component::sceneutility
