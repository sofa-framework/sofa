#ifndef SOFA_COMPONENTS_COMMON_VEC_H
#define SOFA_COMPONENTS_COMMON_VEC_H

/* The following code declares class array,
 * an STL container (as wrapper) for arrays of constant size.
 *
 * See
 *      http://www.josuttis.com/cppcode
 * for details and the latest version.
 *
 * (C) Copyright Nicolai M. Josuttis 2001.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 *
 * 29 Jun 2005 - remove boost includes and reverse iterators. (Jeremie Allard)
 * 23 Aug 2002 - fix for Non-MSVC compilers combined with MSVC libraries.
 * 05 Aug 2001 - minor update (Nico Josuttis)
 * 20 Jan 2001 - STLport fix (Beman Dawes)
 * 29 Sep 2000 - Initial Revision (Nico Josuttis)
 */

// See http://www.boost.org/libs/array for Documentation.

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <algorithm>
#include <math.h>
//#include <boost/static_assert.hpp>
#define BOOST_STATIC_ASSERT(a)

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class T, std::size_t N>
class fixed_array
{
public:
    T elems[N];    // fixed-size array of elements of type T

public:
    // type definitions
    typedef T              value_type;
    typedef T*             iterator;
    typedef const T*       const_iterator;
    typedef T&             reference;
    typedef const T&       const_reference;
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    // iterator support
    iterator begin() { return elems; }
    const_iterator begin() const { return elems; }
    iterator end() { return elems+N; }
    const_iterator end() const { return elems+N; }

    // operator[]
    reference operator[](size_type i) { return elems[i]; }
    const_reference operator[](size_type i) const { return elems[i]; }

    // at() with range check
    reference at(size_type i) { rangecheck(i); return elems[i]; }
    const_reference at(size_type i) const { rangecheck(i); return elems[i]; }

    // front() and back()
    reference front() { return elems[0]; }
    const_reference front() const { return elems[0]; }
    reference back() { return elems[N-1]; }
    const_reference back() const { return elems[N-1]; }

    // size is constant
    static size_type size() { return N; }
    static bool empty() { return false; }
    static size_type max_size() { return N; }
    enum { static_size = N };

    // swap (note: linear complexity)
    void swap (fixed_array<T,N>& y)
    {
        std::swap_ranges(begin(),end(),y.begin());
    }

    // direct access to data
    const T* data() const { return elems; }

    // assignment with type conversion
    template <typename T2>
    fixed_array<T,N>& operator= (const fixed_array<T2,N>& rhs)
    {
        std::copy(rhs.begin(),rhs.end(), begin());
        return *this;
    }

    // assign one value to all elements
    void assign (const T& value)
    {
        std::fill_n(begin(),size(),value);
    }

private:

    // check range (may be private because it is static)
    static void rangecheck (size_type i)
    {
        if (i >= size()) { throw std::range_error("fixed_array"); }
    }

};

// comparisons
template<class T, std::size_t N>
bool operator== (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return std::equal(x.begin(), x.end(), y.begin());
}
template<class T, std::size_t N>
bool operator< (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end());
}
template<class T, std::size_t N>
bool operator!= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return !(x==y);
}
template<class T, std::size_t N>
bool operator> (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return y<x;
}
template<class T, std::size_t N>
bool operator<= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return !(y<x);
}
template<class T, std::size_t N>
bool operator>= (const fixed_array<T,N>& x, const fixed_array<T,N>& y)
{
    return !(x<y);
}

// global swap()
template<class T, std::size_t N>
inline void swap (fixed_array<T,N>& x, fixed_array<T,N>& y)
{
    x.swap(y);
}


template <int N, typename real=float>
class Vec : public fixed_array<real,N>
{
public:

    /// Default constructor: sets all values to 0.
    Vec()
    {
        this->assign(0);
    }

    /*
      Vec(real r1)
      {
        BOOST_STATIC_ASSERT(N == 1);
        this->elems[0]=r1;
      }
    */

    /// Specific constructor for 2-elements vectors.
    Vec(real r1, real r2)
    {
        BOOST_STATIC_ASSERT(N == 2);
        this->elems[0]=r1;
        this->elems[1]=r2;
    }

    /// Specific constructor for 3-elements vectors.
    Vec(real r1, real r2, real r3)
    {
        BOOST_STATIC_ASSERT(N == 3);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
    }

    /// Specific constructor for 4-elements vectors.
    Vec(real r1, real r2, real r3, real r4)
    {
        BOOST_STATIC_ASSERT(N == 4);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
    }

    /// Specific constructor for 5-elements vectors.
    Vec(real r1, real r2, real r3, real r4, real r5)
    {
        BOOST_STATIC_ASSERT(N == 5);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
    }

    /// Specific constructor for 6-elements vectors (bounding-box).
    Vec(real r1, real r2, real r3, real r4, real r5, real r6)
    {
        BOOST_STATIC_ASSERT(N == 6);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
    }

    /// Constructor from an N-1 elements vector and an additional value (added at the end).
    Vec(const Vec<N-1,real>& v, real r1)
    {
        BOOST_STATIC_ASSERT(N > 1);
        for(int i=0; i<N-1; i++)
            this->elems[i] = v[i];
        this->elems[N-1]=r1;
    }

    /// Constructor from an array of values.
    template<typename real2>
    explicit Vec(const real2* p)
    {
        std::copy(p, p+N, this->begin());
    }

    /// Special access to first element.
    real& x() { BOOST_STATIC_ASSERT(N >= 1); return this->elems[0]; }
    /// Special access to second element.
    real& y() { BOOST_STATIC_ASSERT(N >= 2); return this->elems[1]; }
    /// Special access to third element.
    real& z() { BOOST_STATIC_ASSERT(N >= 3); return this->elems[2]; }
    /// Special access to fourth element.
    real& w() { BOOST_STATIC_ASSERT(N >= 4); return this->elems[3]; }

    /// Special const access to first element.
    const real& x() const { BOOST_STATIC_ASSERT(N >= 1); return this->elems[0]; }
    /// Special const access to second element.
    const real& y() const { BOOST_STATIC_ASSERT(N >= 2); return this->elems[1]; }
    /// Special const access to third element.
    const real& z() const { BOOST_STATIC_ASSERT(N >= 3); return this->elems[2]; }
    /// Special const access to fourth element.
    const real& w() const { BOOST_STATIC_ASSERT(N >= 4); return this->elems[3]; }

    /// Assignment operator from an array of values.
    void operator=(const real* p)
    {
        std::copy(p, p+N, this->begin());
    }

    /// Assignment from a vector with different dimensions.
    template<int M, typename real2> void operator=(const Vec<M,real2>& v)
    {
        std::copy(v.begin(), v.begin()+(N>M?M:N), this->begin());
    }

    /// Sets every element to 0.
    void clear()
    {
        this->assign(0);
    }

    /// Sets every element to r.
    void fill(real r)
    {
        this->assign(r);
    }

    /// Access to i-th element.
    real& operator[](int i)
    {
        return this->elems[i];
    }

    /// Const access to i-th element.
    const real& operator[](int i) const
    {
        return this->elems[i];
    }

    /// Access to i-th element.
    real& operator()(int i)
    {
        return this->elems[i];
    }

    /// Const access to i-th element.
    const real& operator()(int i) const
    {
        return this->elems[i];
    }

    /// Cast into a const array of values.
    operator const real*() const
    {
        return this->elems;
    }

    /// Cast into an array of values.
    operator real*()
    {
        return this->elems;
    }

    // LINEAR ALGEBRA

    /// Multiplication by a scalar f.
    Vec<N,real> operator*(real f) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]*f;
        return r;
    }

    /// On-place multiplication by a scalar f.
    void operator*=(real f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]*=f;
    }

    /// Division by a scalar f.
    Vec<N,real> operator/(real f) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]/f;
        return r;
    }

    /// On-place division by a scalar f.
    void operator/=(real f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]/=f;
    }

    /// Dot product.
    real operator*(const Vec<N,real>& v) const
    {
        real r = this->elems[0]*v[0];
        for (int i=1; i<N; i++)
            r += this->elems[i]*v[i];
        return r;
    }

    /// Vector addition.
    Vec<N,real> operator+(const Vec<N,real>& v) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]+v[i];
        return r;
    }

    /// On-place vector addition.
    template<class real2>
    void operator+=(const Vec<N,real2>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]+=v[i];
    }

    /// Vector subtraction.
    Vec<N,real> operator-(const Vec<N,real>& v) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]-v[i];
        return r;
    }

    /// On-place vector subtraction.
    void operator-=(const Vec<N,real>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]-=v[i];
    }

    /// Vector negation.
    Vec<N,real> operator-() const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i]=-this->elems[i];
        return r;
    }

    /// Squared norm.
    real norm2() const
    {
        real r = this->elems[0]*this->elems[0];
        for (int i=1; i<N; i++)
            r += this->elems[i]*this->elems[i];
        return r;
    }

    /// Euclidean norm.
    real norm() const
    {
        return sqrt(norm2());
    }

    /// Normalize the vector.
    void normalize()
    {
        real r = norm();
        if (r>1e-10)
            for (int i=0; i<N; i++)
                this->elems[i]/=r;
    }

};

/// Cross product for 3-elements vectors.
template<typename real>
inline Vec<3,real> cross(const Vec<3,real>& a, const Vec<3,real>& b)
{
    return Vec<3,real>(a.y()*b.z() - a.z()*b.y(),
            a.z()*b.x() - a.x()*b.z(),
            a.x()*b.y() - a.y()*b.x());
}

/// Dot product (alias for operator*)
template<int N,typename real>
inline real dot(const Vec<N,real>& a, const Vec<N,real>& b)
{
    return a*b;
}

typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;

typedef Vec3d Vector3; ///< alias

#undef BOOST_STATIC_ASSERT

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
