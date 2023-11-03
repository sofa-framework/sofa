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

#include <sofa/type/config.h>

#include <vector>
#include <string>
#include <typeinfo>
#include <istream>
#include <ostream>

#if !defined(NDEBUG) && !defined(SOFA_NO_VECTOR_ACCESS_FAILURE)
#define SOFA_VECTOR_CHECK_ACCESS true
#else
#define SOFA_VECTOR_CHECK_ACCESS false
#endif

namespace sofa::type
{

static constexpr bool isEnabledVectorAccessChecking {SOFA_VECTOR_CHECK_ACCESS};

[[noreturn]]
extern void SOFA_TYPE_API vector_access_failure(const void* vec, std::size_t size, std::size_t i, const std::type_info& type);

// standard vector dont use the CPUMemoryManager given as template
template <typename T>
class CPUMemoryManager;

/// Regular vector
/// Using CPUMemoryManager, it has the same behavior as std::vector with extra conveniences:
///  - string serialization (making it usable in Data)
///  - operator[] is checking if the index is within the bounds in debug
template <class T, class MemoryManager = CPUMemoryManager<T>>
class vector : public std::vector<T, std::allocator<T> >
{
public:
    typedef CPUMemoryManager<T> memory_manager;
    typedef std::allocator<T> Alloc;
    /// Size
    typedef typename std::vector<T,Alloc>::size_type Size;
    /// reference to a value (read-write)
    typedef typename std::vector<T,Alloc>::reference reference;
    /// const reference to a value (read only)
    typedef typename std::vector<T,Alloc>::const_reference const_reference;

    template<class T2>
    using rebind_to = vector< T2, CPUMemoryManager<T2> >;

    /// Basic constructor
    vector() : std::vector<T,Alloc>() {}
    /// Constructor
    vector(Size n, const T& value): std::vector<T,Alloc>(n,value) {}
    /// Constructor
    explicit vector(Size n): std::vector<T,Alloc>(n) {}
    /// Constructor
    vector(const std::vector<T, Alloc>& x): std::vector<T,Alloc>(x) {}
    /// Brace initalizer constructor
    vector(const std::initializer_list<T>& t) : std::vector<T,Alloc>(t) {}
    /// Move constructor
    vector(std::vector<T,Alloc>&& v): std::vector<T,Alloc>(std::move(v)) {}

    /// Copy operator
    vector& operator=(const std::vector<T, Alloc>& x)
    {
        std::vector<T,Alloc>::operator=(x);
        return *this;
    }
    /// Move assignment operator
    vector& operator=(std::vector<T,Alloc>&& v)
    {
        std::vector<T,Alloc>::operator=(std::move(v));
        return *this;
    }

    /// Constructor
    vector(typename vector<T>::const_iterator first, typename vector<T>::const_iterator last): std::vector<T>(first,last) {}

    /// Read/write random access
    reference operator[](Size n)
    {
        if constexpr (sofa::type::isEnabledVectorAccessChecking)
        {
            if (n >= this->size())
                vector_access_failure(this, this->size(), n, typeid(T));
        }
        return std::vector<T>::operator[](n);
    }

    /// Read-only random access
    const_reference operator[](Size n) const
    {
        if constexpr (sofa::type::isEnabledVectorAccessChecking)
        {
            if (n >= this->size())
                vector_access_failure(this, this->size(), n, typeid(T));
        }
        return std::vector<T>::operator[](n);
    }

    std::ostream& write(std::ostream& os) const
    {
        if( this->size()>0 )
        {
            for( Size i=0; i<this->size()-1; ++i )
                os<<(*this)[i]<<" ";
            os<<(*this)[this->size()-1];
        }
        return os;
    }

    std::istream& read(std::istream& in)
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


    /// Output stream
    friend std::ostream& operator<< ( std::ostream& os, const vector& vec ) { return vec.write(os); }

    /// Input stream
    friend std::istream& operator>> ( std::istream& in, vector& vec ){ return vec.read(in); }

    /// Sets every element to 'value'
    void fill( const T& value )
    {
        std::fill(this->begin(), this->end(), value);
    }

    /// this function is usefull for vector_device because it resize the vector without device operation (if device is not valid).
    /// Therefore the function is used in asynchronous code to safly resize a vector which is either cuda of type::vector
    void fastResize(Size n)
    {
        this->resize(n);
    }

};

} /// namespace sofa::type
