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
#include <sofa/linearalgebra/config.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/linearalgebra/BaseVector.h>

namespace sofa::linearalgebra
{

#if !defined(SOFA_NO_VECTOR_ACCESS_FAILURE) && !defined(NDEBUG)
#define DO_CHECK_VECTOR_ACCESS true
#else
#define DO_CHECK_VECTOR_ACCESS false
#endif ///

template<typename T>
class FullVector : public linearalgebra::BaseVector
{
public:
    typedef T Real;
    typedef linearalgebra::BaseVector::Index Index;
    typedef T* Iterator;
    typedef const T* ConstIterator;

    typedef Real value_type;
    typedef Index Size;
    typedef Iterator iterator;
    typedef ConstIterator const_iterator;

protected:
    T* data;
    Index cursize;
    Index allocsize;

    void checkIndex(Index n) const;

public:

    FullVector()
        : linearalgebra::BaseVector()
        , data(nullptr), cursize(0), allocsize(0)
    {
    }

    FullVector(const FullVector& vect)
        : linearalgebra::BaseVector()
        , data(nullptr), cursize(0), allocsize(0)
    {
        (*this) = vect;
    }

    explicit FullVector(Index n)
        : linearalgebra::BaseVector()
        , data(new T[n]), cursize(n), allocsize(n)
    {
    }

    FullVector(T* ptr, Index n)
        : linearalgebra::BaseVector()
        , data(ptr), cursize(n), allocsize(-n)
    {
    }

    FullVector(T* ptr, Index n, Index nmax)
        : linearalgebra::BaseVector()
        , data(ptr), cursize(n), allocsize(-nmax)
    {
    }

    ~FullVector() override
    {
        if (allocsize>0)
            delete[] data;
    }

    T* ptr() { return data; }
    const T* ptr() const { return data; }

    void setptr(T* p) { data = p; }

    Index capacity() const { if (allocsize < 0) return -allocsize; else return allocsize; }

    Iterator begin() { return data; }
    Iterator end()   { return data+cursize; }

    ConstIterator begin() const { return data; }
    ConstIterator end()   const { return data+cursize; }

    void fastResize(Index dim);

    void resize(Index dim) override;
    void clear() override;

    void swap(FullVector<T>& v);

    // for compatibility with baseVector
    void clear(Index dim) override;

    T& operator[](Index i)
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
            checkIndex(i);
        return data[i];
    }

    const T& operator[](Index i) const
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
            checkIndex(i);
        return data[i];
    }

    SReal element(Index i) const override
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
            checkIndex(i);
        return (SReal) data[i];
    }

    void set(Index i, SReal v) override
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
            checkIndex(i);
        data[i] = (Real)v;
    }

    void add(Index i, SReal v) override
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
            checkIndex(i);
        data[i] +=  (Real)v;
    }

    Index size() const override
    {
        return cursize;
    }

    FullVector<T> sub(Index i, Index n)
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
        {
            if (n > 0)
                checkIndex(i+n-1);
        }
        return FullVector<T>(data+i,n);
    }

    template<class TV>
    void getsub(Index i, Index n, TV& v)
    {
        if constexpr(DO_CHECK_VECTOR_ACCESS)
        {
            if (n > 0)
                checkIndex(i+n-1);
        }
        v = FullVector<T>(data+i,n);
    }

    template<class TV>
    void setsub(Index i, Index n, const TV& v)
    {
        if (n > 0) checkIndex(i+n-1);
        FullVector<T>(data+i,n) = v;
    }

    /// v = a
    void operator=(const FullVector<T>& a);

    void operator=(const T& a);

    /// v += a
    void operator+=(const FullVector<Real>& a);

    /// v -= a
    void operator-=(const FullVector<Real>& a);

    /// v = a*f
    void eq(const FullVector<Real>& a, Real f);

    /// v = a+b*f
    void eq(const FullVector<Real>& a, const FullVector<Real>& b, Real f=1.0);

    /// v += a*f
    void peq(const FullVector<Real>& a, Real f);

    /// v *= f
    void operator*=(Real f);

    /// \return v.a
    Real dot(const FullVector<Real>& a) const;

    /// \return sqrt(v.v)
    double norm() const;

    static const char* Name() { return "FullVector"; }
};

SOFA_LINEARALGEBRA_API std::ostream& operator <<(std::ostream& out, const FullVector<float>& v);
SOFA_LINEARALGEBRA_API std::ostream& operator <<(std::ostream& out, const FullVector<double>& v);

#if !defined(SOFA_LINEARALGEBRA_FULLVECTOR_DEFINITION)
extern template class SOFA_LINEARALGEBRA_API FullVector<float>;
extern template class SOFA_LINEARALGEBRA_API FullVector<double>;
#endif


} // namespace sofa::linearalgebra
