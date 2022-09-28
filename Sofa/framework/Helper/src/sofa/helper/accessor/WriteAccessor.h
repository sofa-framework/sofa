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

#include <sofa/helper/accessor/WriteAccessorVector.h>

namespace sofa::helper
{


/** A WriteAccessor is a proxy class, holding a reference to a given container
 *  and providing access to its data, using an unified interface (similar to
 *  std::vector), hiding API differences within some containers.
 *
 *  Other advantadges of using a WriteAccessor are :
 *
 *  - It can be faster that the default methods and operators of the container,
 *  as verifications and changes notifications can be handled in the accessor's
 *  constructor and destructor instead of at each item access.
 *
 *  - Accesses can be logged for debugging or task dependencies analysis.
 *
 *  The default implementation provides only minimal set of methods and
 *  operators, sufficient for scalar types but which should be overloaded for
 *  more complex types.
 *  Various template specializations are typically used, especially for core::objectmodel::Data<T>
 */
template<class T, class Enable = void>
class WriteAccessor
{
public:
    static_assert(!std::is_const_v<T>, "Trying to have write access on a const type");

    typedef T container_type;
    typedef T value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef const value_type& const_reference;
    typedef const value_type* const_pointer;

protected:
    container_type* vref;

public:
    explicit WriteAccessor(container_type& container) : vref(&container) {}

    const_reference ref() const { return *vref; }
    reference wref() { return *vref; }

    operator  const_reference () const { return  *vref; }
    const_pointer   operator->() const { return vref; }
    const_reference operator* () const { return  *vref; }

    operator  reference () { return  *vref; }
    pointer   operator->() { return vref; }
    reference operator* () { return  *vref; }

    template<class T2>
    void operator=(const T2& v)
    {
        vref = &v;
    }
};

template<class VectorLikeType>
class WriteAccessor<VectorLikeType,
                    std::enable_if_t<sofa::type::trait::is_vector<VectorLikeType>::value>>
    : public WriteAccessorVector< VectorLikeType >
{
public:
    typedef WriteAccessorVector< VectorLikeType > Inherit;
    typedef typename Inherit::container_type container_type;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

}
