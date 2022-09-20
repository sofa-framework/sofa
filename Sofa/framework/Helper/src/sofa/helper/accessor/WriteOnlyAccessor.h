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

#include <sofa/helper/accessor/WriteAccessor.h>

namespace sofa::helper
{

/** Identical to WriteAccessor for default implementation, but different for some template specializations such as  core::objectmodel::Data<T>
*/
template<class T, class Enable = void>
class WriteOnlyAccessor : public WriteAccessor<T, Enable>
{
protected:
    typedef WriteAccessor<T> Inherit;
    typedef typename Inherit::container_type container_type;

public:
    explicit WriteOnlyAccessor(container_type& container) : WriteAccessor<T, Enable>(container) {}
};

template<class VectorLikeType>
class WriteOnlyAccessor<VectorLikeType,
                        std::enable_if_t<sofa::type::trait::is_vector<VectorLikeType>::value> >
    : public WriteAccessorVector< VectorLikeType >
{
public:
    typedef WriteAccessorVector< VectorLikeType > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteOnlyAccessor(container_type& c) : Inherit(c) {}
};


}
