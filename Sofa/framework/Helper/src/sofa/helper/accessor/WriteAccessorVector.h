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

    ////// Capacity //////    
    bool empty() const { return vref->empty(); }
    Size size() const { return vref->size(); }
    void reserve(Size s) { vref->reserve(s); }

    ////// Element access //////    
    reference operator[](Size pos) { return (*vref)[pos]; }
    const_reference operator[](Size pos) const { return (*vref)[pos]; }

    reference front() { return vref->front(); }
    const_reference front() const { return vref->front(); }

    reference back() { return vref->back(); }
    const_reference back() const { return vref->back(); }


    ////// Iterators //////    
    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }


    ////// Modifiers //////    
    void clear() { vref->clear(); }
    SOFA_WRITEACCESSOR_RESIZE_DISABLED() void resize(Size s, bool) { vref->resize(s); }
    void resize(Size s) { vref->resize(s); }
    
    iterator insert(const_iterator pos, const T& value) { return vref->insert(pos, value); }
    iterator erase(iterator pos) { return vref->erase(pos); }
    iterator erase(const_iterator pos) { return vref->erase(pos); }

    void push_back(const value_type& v) { vref->push_back(v); }
    template <class... Args>
    reference emplace_back(Args&&... args) { return vref->emplace_back(std::forward<Args>(args)...);}
    void pop_back() { vref->pop_back(); }


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
