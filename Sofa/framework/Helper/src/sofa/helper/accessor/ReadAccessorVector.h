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

#include <sofa/type/trait/is_vector.h>
#include <iosfwd>

namespace sofa::helper
{
////////////////////////// ReadAccessor for wrapping around vector like object //////////////////////
/// ReadAccessor implementation class for vector types
template<class T>
class ReadAccessorVector
{
public:
    typedef T container_type;
    typedef const T const_container_type;
    typedef typename container_type::Size Size;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    const container_type* vref;

public:
    ReadAccessorVector(const container_type& container) : vref(&container) {}

    bool empty() const { return vref->empty(); }
    Size size() const { return vref->size(); }
    const_reference operator[](Size i) const { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    const_iterator end() const { return vref->end(); }

    ///////// Access the container for reading ////////////////
    operator const_container_type& () const { return  *vref; }
    const_container_type* operator->() const { return vref; }
    const_container_type& operator* () const { return  *vref; }
    const_container_type& ref() const { return *vref; }          ///< this duplicate operator* (remove ?)
    ///////////////////////////////////////////////////////////
};

}
