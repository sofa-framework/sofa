/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/objectmodel/BaseData.h>

namespace sofa::core::objectmodel
{


DataLink::DataLink(BaseData& owner) :
    m_owner(owner)
  , m_dest(nullptr)
{
}

DataLink::~DataLink(){}

void DataLink::unSet()
{
    assert(isSet());

    if(!isSet())
        return;

    /// First retrieve from the parent the last know value.
    m_dest->updateIfDirty();

    /// Then disconnect the dependency graph
    m_owner.getDDGNode()->delInput(m_dest->getDDGNode());

    /// Finalize the unsetting.
    m_dest=nullptr;
}

void DataLink::set(BaseData* dest)
{
    /// Disconnect the previous data link.
    if(m_dest)
        m_owner.getDDGNode()->delInput(m_dest->getDDGNode());
    m_dest=dest;

    /// Connect the new data link.
    m_owner.getDDGNode()->addInput(m_dest->getDDGNode());
}

} ///namespace sofa::core::objectmodel
