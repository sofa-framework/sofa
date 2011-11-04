/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_BASEVECTOR_H
#define SOFA_DEFAULTTYPE_BASEVECTOR_H

#include <sofa/helper/system/config.h>
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
    virtual ~BaseVector() {}

    /// Number of elements
    virtual unsigned int size(void) const = 0;
    /// Read the value of element i
    virtual SReal element(int i) const = 0;

    /// Resize the vector, and reset all values to 0
    virtual void resize(int dim) = 0;
    /// Reset all values to 0
    virtual void clear() = 0;

    /// Write the value of element i
    virtual void set(int i, SReal v) = 0;
    /// Add v to the existing value of element i
    virtual void add(int i, SReal v) = 0;

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
    virtual unsigned int getElementSize() const { return sizeof(SReal); }

    /// Return true if this vector is full, i.a. all elements are stored in memory
    virtual bool isFull() const { return true; }

    /// Return true if this vector is sparse, i.a. only some of the elements are stored in memory.
    /// This is the exact opposite to isFull().
    bool isSparse() const { return !isFull(); }

    /// @}

protected:

    template<class T>
    const T* elementsDefaultImpl(int i0, int n, T* buffer) const
    {
        if (buffer)
            for (int i=0; i<n; ++i)
                buffer[i]=(T)element(i0+i);
        return buffer;
    }

    template<class T>
    void setDefaultImpl(int i0, int n, const T* src)
    {
        for (int i=0; i<n; ++i)
            set(i0+i,(SReal)src[i]);
    }

    template<class T>
    void addDefaultImpl(int i0, int n, const T* src)
    {
        for (int i=0; i<n; ++i)
            add(i0+i,(SReal)src[i]);
    }

public:

    /// Get the values of n elements, starting at element i0, into given float buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const float* elements(int i0, int n, float* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Get the values of n elements, starting at element i0, into given double buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const double* elements(int i0, int n, double* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Get the values of n elements, starting at element i0, into given int buffer, or return the pointer to the data if the in-memory format is compatible
    virtual const int* elements(int i0, int n, int* src) const
    {
        return elementsDefaultImpl(i0,n,src);
    }

    /// Write the values of n float elements, starting at element i0
    virtual void set(int i0, int n, const float* src)
    {
        setDefaultImpl(i0,n,src);
    }

    /// Write the values of n double elements, starting at element i0
    virtual void set(int i0, int n, const double* src)
    {
        setDefaultImpl(i0,n,src);
    }

    /// Write the values of n int elements, starting at element i0
    virtual void set(int i0, int n, const int* src)
    {
        setDefaultImpl(i0,n,src);
    }


    /// Add to the values of n float elements, starting at element i0
    virtual void add(int i0, int n, const float* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /// Add to the values of n double elements, starting at element i0
    virtual void add(int i0, int n, const double* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /// Add to the values of n int elements, starting at element i0
    virtual void add(int i0, int n, const int* src)
    {
        addDefaultImpl(i0,n,src);
    }

    /*
        /// Write the value of element i
        virtual void set(int i, SReal v) { set(i,(SReal)v); }
        /// Add v to the existing value of element i
        virtual void add(int i, SReal v) { add(i,(SReal)v); }
    */
    /// Reset the value of element i to 0
    virtual void clear(int i) { set(i,0.0); }
};

} // nampespace defaulttype

} // nampespace sofa


#endif
