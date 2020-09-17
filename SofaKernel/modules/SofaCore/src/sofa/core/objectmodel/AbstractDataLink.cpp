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
const BaseData& AbstractDataLink::getOwner() { return __doGetOwner__(); }

/// Change the targetted DataField
void AbstractDataLink::setTarget(BaseData* target){ __doSetTarget__(target); }

/// Get the targetted DataField
BaseData* AbstractDataLink::getTarget(){ return __doGetTarget__(); }

const std::string AbstractDataLink::getPath() const
{
    return m_path;
}

void AbstractDataLink::setPath(const std::string& path)
{
    /// Trying to resolve link
    m_path = path;
    resolvePathAndSetData();
}

bool AbstractDataLink::resolvePathAndSetData()
{
    BaseData *data = PathResolver::FindBaseDataFromPath(&getOwner(), getPath());
    if(data == nullptr)
        return false;
    setTarget(data);
    return true;
}

bool AbstractDataLink::hasPath() const { return !m_path.empty(); }

}
