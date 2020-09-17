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

#include <string>
#include <sofa/core/config.h>
#include <sofa/core/objectmodel/AbstractDataLink.h>

namespace sofa::core::objectmodel
{

/**
 * @brief DataLink stores a connection between two object of type Data<XXX>
 * The class is templated by the Data type to connect.
 * The class implements the AbstractDataLink interface.
 */
template<class T>
class SOFA_CORE_API DataLink final : public AbstractDataLink
{
public:

    DataLink(T& owner) : m_owner{owner} { }
    virtual ~DataLink() {}

    T* getTarget()
    {
        if(m_target==nullptr && !m_path.empty())
            resolvePathAndSetData();
        return m_target;
    }

    void unSet(){ m_target=nullptr; m_path = ""; }
    bool isSet() const { return m_target != nullptr; }
    void setTarget(T* target)
    {
        m_path = "";
        m_target = target;
    }

    T& getOwner() const { return m_owner; }

protected:
    /// Take the "generic" data and cast it to the expected type.
    void __doSetTarget__(BaseData* target) override
    {
        setTarget(dynamic_cast<T*>(target));
    }

    /// Returns the typed data to its abstract one
    BaseData* __doGetTarget__() override
    {
        return DataLink::getTarget();
    }

    /// Returns the typed data to its abstract one.
    const BaseData& __doGetOwner__() override
    {
        return DataLink::getOwner();
    }

private:
    T& m_owner  ;
    T* m_target {nullptr};
};

} /// namespace sofa::core::objectmodel

