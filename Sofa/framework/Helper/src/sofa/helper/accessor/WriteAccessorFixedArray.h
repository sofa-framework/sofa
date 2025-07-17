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

#include <sofa/helper/config.h>

namespace sofa::helper
{

/// WriteAccessor implementation class for fixed array types
template<type::trait::is_fixed_array T>
class WriteAccessorFixedArray
{
public:
    typedef T container_type;
    typedef const T const_container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename container_type::reference reference;
    typedef typename container_type::const_reference const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

protected:
    container_type* vref;

public:
    WriteAccessorFixedArray(container_type& container) : vref(&container) {}

    ////// Capacity //////
    bool empty() const { return false; }
    size_type size() const { return T::static_size; }

    ////// Element access //////
    reference operator[](size_type pos) { return (*vref)[pos]; }
    const_reference operator[](size_type pos) const { return (*vref)[pos]; }

    reference front() { return vref->front(); }
    const_reference front() const { return vref->front(); }

    reference back() { return vref->back(); }
    const_reference back() const { return vref->back(); }


    ////// Iterators //////
    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }
    
    ////// Access the container in reading & writing //////
    operator  container_type () { return  *vref; }
    container_type* operator->() { return vref; }
    container_type& operator* () { return  *vref; }
    container_type& wref() { return *vref; }
    ///////////////////////////////////////////////////////

    ///////// Access the container for reading ////////////////
    operator  const_container_type () const { return  *vref; }
    const_container_type* operator->() const { return vref; }
    const_container_type& operator* () const { return  *vref; }

    /// this one duplicate operator*
    const container_type& ref() const { return *vref; }
    ///////////////////////////////////////////////////////////
};

}
