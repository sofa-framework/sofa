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
#pragma once

#include <cassert>
#include <sofa/core/core.h>

namespace sofa::core::objectmodel
{

class BaseData;
class DDGNode;

/**
 *  \brief Store a link between Data.
 *  If this happens an edge is added in the DDG.
 */
class SOFA_CORE_API DataLink
{
public:
    DataLink(BaseData& owner);
    virtual ~DataLink();

    BaseData& m_owner;
    BaseData* m_dest {nullptr};

    void set(BaseData* dest);

    BaseData& getOwner() const {return m_owner;}
    BaseData* get() const {return m_dest;}

    bool m_isPersistant {false};

    bool isSet(){ return m_dest != nullptr; }
    void unSet();

    void setPersistent(bool b) { m_isPersistant = b; }
    bool isPersistent() const { return m_isPersistant; }

    // DDGNode* m_ddgnode;
};

} /// namespace sofa::core::objectmodel

