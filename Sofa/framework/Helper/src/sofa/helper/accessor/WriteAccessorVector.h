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

/// WriteAccessor implementation class for vector types
template<class T>
class WriteAccessorVector
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
    container_type* vref;

public:
    WriteAccessorVector(container_type& container) : vref(&container) {}

    bool empty() const { return vref->empty(); }
    Size size() const { return vref->size(); }

    const_reference operator[](Size i) const { return (*vref)[i]; }
    reference operator[](Size i) { return (*vref)[i]; }

    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }

    void clear() { vref->clear(); }
    SOFA_WRITEACCESSOR_RESIZE_DEPRECATED() void resize(Size s, bool) { vref->resize(s); }
    void resize(Size s) { vref->resize(s); }
    void reserve(Size s) { vref->reserve(s); }
    void push_back(const value_type& v) { vref->push_back(v); }
    template <class... Args>
    reference emplace_back(Args&&... args) { return vref->emplace_back(std::forward<Args>(args)...);}

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
