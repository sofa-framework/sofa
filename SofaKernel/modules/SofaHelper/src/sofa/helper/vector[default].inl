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
#include <cassert>
#include <iostream>
#include<sofa/helper/vector[default].h>

namespace sofa::helper
{

template<class T, class MemoryManager>
void vector<T, MemoryManager>::fill( const T& value )
{
    std::fill( this->begin(), this->end(), value );
}

/// this function is usefull for vector_device because it resize the vector without device operation (if device is not valid).
/// Therefore the function is used in asynchronous code to safly resize a vector which is either cuda of helper::vector
template<class T, class MemoryManager>
void vector<T, MemoryManager>::fastResize(Size n)
{
    this->resize(n);
}

/// Read/write random access
template<class T, class MemoryManager>
typename vector<T, MemoryManager>::reference vector<T, MemoryManager>::operator[](Size n)
{
    if constexpr(sofa::helper::isEnabledVectorAccessChecking)
    {
        if( n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
    }
    return *(this->begin() + n);
}

/// Read-only random access
template<class T, class MemoryManager>
typename vector<T, MemoryManager>::const_reference vector<T, MemoryManager>::operator[](Size n) const
{
    if constexpr (sofa::helper::isEnabledVectorAccessChecking)
    {
        if (n>=this->size())
            vector_access_failure(this, this->size(), n, typeid(T));
    }
    return *(this->begin() + n);
}

template<class T, class MemoryManager>
std::ostream& vector<T, MemoryManager>::write(std::ostream& os) const
{
    if( this->size()>0 )
    {
        for( Size i=0; i<this->size()-1; ++i )
            os<<(*this)[i]<<" ";
        os<<(*this)[this->size()-1];
    }
    return os;
}

template<class T, class MemoryManager>
std::istream& vector<T, MemoryManager>::read(std::istream& in)
{
    T t=T();
    this->clear();
    while(in>>t)
    {
        this->push_back(t);
    }
    if( in.rdstate() & std::ios_base::eofbit ) { in.clear(); }
    return in;
}

}
