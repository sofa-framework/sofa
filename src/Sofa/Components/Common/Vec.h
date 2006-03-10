#ifndef SOFA_COMPONENTS_COMMON_VEC_H
#define SOFA_COMPONENTS_COMMON_VEC_H

#include "fixed_array.h"
#include <math.h>
//#include <boost/static_assert.hpp>
#define BOOST_STATIC_ASSERT(a)

namespace Sofa
{

namespace Components
{

namespace Common
{

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

    /// Constructor from an N-1 elements vector and an additional value (added at the end).
    Vec(const Vec<N-1,real>& v, real r1)
    {
        BOOST_STATIC_ASSERT(N > 1);
        for(int i=0; i<N-1; i++)
            this->elems[i] = v[i];
        this->elems[N-1]=r1;
    }

    template<typename real2>
    Vec(const Vec<N, real2>& p)
    {
        std::copy(p.begin(), p.end(), this->begin());
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
