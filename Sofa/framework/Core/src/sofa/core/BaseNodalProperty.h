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

#include <sofa/core/objectmodel/BaseComponent.h>

namespace sofa::core
{

template<class T>
class BaseNodalProperty : public virtual sofa::core::objectmodel::BaseComponent
{
public:
    SOFA_CLASS(BaseNodalProperty<T>, sofa::core::objectmodel::BaseComponent);

    BaseNodalProperty() = delete;

    Data<sofa::type::vector<T> > d_property;

    const T& getNodeProperty(sofa::Index i, sofa::helper::ReadAccessor<Data<sofa::type::vector<T>>>& property) const
    {
        if (property.size() > i)
        {
            return property[i];
        }
        if (!property.empty())
        {
            return property->back();
        }
        return m_defaultProperty;
    }

    const T& getNodeProperty(sofa::Index i) const
    {
        sofa::helper::ReadAccessor property { d_property };
        return getNodeProperty(i, property);
    }

protected:
    explicit BaseNodalProperty(const T defaultProperty)
        : d_property(initData(&d_property, {defaultProperty}, "property", "Nodal property"))
        , m_defaultProperty(defaultProperty)
    {}

    T m_defaultProperty {};
};

}
