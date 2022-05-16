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
#include <sofa/core/objectmodel/DataLink.h>
#include <sofa/core/PathResolver.h>
using sofa::core::PathResolver;

namespace sofa::core::objectmodel
{

/// Get the DataField having thins link as an attribute
/// there is a one to one owner relationship.
const BaseData& AbstractDataLink::getOwner() const { return _doGetOwner_(); }

/// Change the targetted DataField
void AbstractDataLink::setTarget(BaseData* target){ _doSetTarget_(target); }

/// Get the targetted DataField
BaseData* AbstractDataLink::getTarget() const
{
    return _doGetTarget_();
}

const std::string AbstractDataLink::getPath() const
{
    return m_path;
}

bool AbstractDataLink::hasTarget() const
{
    return _doGetTarget_() != nullptr;
}

void AbstractDataLink::setPath(const std::string& path)
{
    /// Trying to resolve link
    m_path = path;
    resolvePathAndSetTarget();
}

bool AbstractDataLink::resolvePathAndSetTarget()
{
    if(getPath().empty())
        return false;

    BaseData *data = PathResolver::FindBaseDataFromPath(&getOwner(), getPath());
    if(data == nullptr)
        return false;
    setTarget(data);
    return true;
}

BaseData* AbstractDataLink::resolvePathAndGetTarget()
{
    BaseData *data = _doGetTarget_();
    if(data!=nullptr)
        return data;


    if(getPath().empty())
        return nullptr;

    data = PathResolver::FindBaseDataFromPath(&getOwner(), getPath());
    if(data == nullptr)
        return nullptr;
    setTarget(data);
    return data;

}

bool AbstractDataLink::hasPath() const { return !m_path.empty(); }

}
