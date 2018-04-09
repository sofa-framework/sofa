/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_DEFAULTTYPE_BASEVECTOR_H
#define SOFA_DEFAULTTYPE_BASEVECTOR_H

#include <sofa/helper/system/config.h>
#include <iostream>

namespace sofa
{

namespace defaulttype
{

/// Generic vector API, allowing to fill and use a vector independently of the linear algebra library in use.
///
/// Note that accessing values using this class is rather slow and should only be used in codes where the
/// provided genericity is necessary.
class BaseVector
{
public:
    typedef int Index;

    virtual ~BaseVector() {}

    /// Number of elements
    virtual Index size(void) const = 0;
    /// Read the value of element i
    virtual SReal element(Index i) const = 0;

    /// Resize the vector, and reset all values to 0
    virtual void resize(Index dim) = 0;
    /// Reset all values to 0
    virtual void clear() = 0;

    /// Write the value of element i
    virtual void set(Index i, SReal v) = 0;
    /// Add v to the existing value of element i
    virtual void add(Index i, SReal v) = 0;

    /// @name Get information about the content and structure of this vector
    /// @{

    enum ElementType
    {
        ELEMENT_UNKNOWN = 0,
        ELEMENT_FLOAT,
        ELEMENT_INT,
    };

    /// @return type of elements stored in this matrix
    virtual ElementType getElementType() const { return ELEMENT_FLOAT; }

    /// @return size of elements stored in this matrix
    virtual std::size_t getElementSize() const { return sizeof(SReal); }

    /// Return true if this vector is full, i.a. all elements are stored in memory
    virtual bool isFull() const { return true; }

    /// Return true if this vector is sparse, i.a. only some of the elements are stored in memory.
    /// This is the exact opposite to isFull().
    bool isSparse() const { return !isFull(); }

    /// @}

protected:

    template<class T>
    const T* elementsDefaultImpl(Index i0, Index n, T* buffer) const
    {
        if (buffer)
            for (Index i=0; i<n; ++i)
                buffer[i]=(T)element(i0+i);
        return buffer;
    }

    template<class T>
    void setDefaultImpl(Index i0, Index n, const T* src)
    {
        for (Index i=0; i<n; ++i)
            set(i0+i,(SReal)src[i]);
    }

    template<class T>
    void addDefaultImpl(Index i0, Index n, const T* src)
    {
        for (Index i=0; i<n; ++i)
            add(i0+i,(SReal)src[i]);
    }

public:

    /// Get the values of n elements, starting at element i0, into given float buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const float* elements(Index i0, Index n, float* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Get the values of n elements, starting at element i0, into given double buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const double* elements(Index i0, Index n, double* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Get the values of n elements, starting at element i0, into given int buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const int* elements(Index i0, Index n, int* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Write the values of n float elements, starting at element i0
    virtual void set(Index i0, Index n, const float* src)
    {
        setDefaultImpl(i0,n,src);
    }

    /// Write the values of n double elements, starting at element i0
    virtual void set(Index i0, Index n, const double* src)
    {
        setDefaultImpl(i0,n,src);
    }

    /// Write the values of n int elements, starting at element i0
    virtual void set(Index i0, Index n, const int* src)
    {
        setDefaultImpl(i0,n,src);
    }


    /// Add to the values of n float elements, starting at element i0
    virtual void add(Index i0, Index n, const float* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /// Add to the values of n double elements, starting at element i0
    virtual void add(Index i0, Index n, const double* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /// Add to the values of n int elements, starting at element i0
    virtual void add(Index i0, Index n, const int* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /*
        /// Write the value of element i
        virtual void set(Index i, SReal v) { set(i,(SReal)v); }
        /// Add v to the existing value of element i
        virtual void add(Index i, SReal v) { add(i,(SReal)v); }
    */
    /// Reset the value of element i to 0
    virtual void clear(Index i) { set(i,0.0); }

    friend std::ostream& operator << (std::ostream& out, const BaseVector& v )
    {
        Index ny = v.size();
        for (Index y=0; y<ny; ++y)
        {
            out << " " << v.element(y);
        }
        return out;
    }

};

} // nampespace defaulttype

} // nampespace sofa


#endif
