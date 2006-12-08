#ifndef SOFA_COMPONENTS_COMMON_VEC_H
#define SOFA_COMPONENTS_COMMON_VEC_H

#include "fixed_array.h"
#include "static_assert.h"
#include <math.h>
#include <functional>
//#define BOOST_STATIC_ASSERT(a)

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

    /// Specific constructor for 1-element vectors.
    Vec(real r1)
    {
        BOOST_STATIC_ASSERT(N == 1);
        this->elems[0]=r1;
    }

    /// Specific constructor for 1-element vectors.
    void operator=(real r1)
    {
        BOOST_STATIC_ASSERT(N == 1);
        this->elems[0]=r1;
    }

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

    /// Specific constructor for 6-elements vectors.
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

    /// Specific constructor for 8-elements vectors.
    Vec(real r1, real r2, real r3, real r4, real r5, real r6, real r7, real r8)
    {
        BOOST_STATIC_ASSERT(N == 8);
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
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
        //std::copy(p.begin(), p.end(), this->begin());
        for(int i=0; i<N; i++)
            this->elems[i] = (real)p(i);
    }

    /// Constructor from an array of values.
    template<typename real2>
    explicit Vec(const real2* p)
    {
        //std::copy(p, p+N, this->begin());
        for(int i=0; i<N; i++)
            this->elems[i] = (real)p[i];
    }

    /// Special access to first element.
    real& x()
    {
        BOOST_STATIC_ASSERT(N >= 1);
        return this->elems[0];
    }
    /// Special access to second element.
    real& y()
    {
        BOOST_STATIC_ASSERT(N >= 2);
        return this->elems[1];
    }
    /// Special access to third element.
    real& z()
    {
        BOOST_STATIC_ASSERT(N >= 3);
        return this->elems[2];
    }
    /// Special access to fourth element.
    real& w()
    {
        BOOST_STATIC_ASSERT(N >= 4);
        return this->elems[3];
    }

    /// Special const access to first element.
    const real& x() const
    {
        BOOST_STATIC_ASSERT(N >= 1);
        return this->elems[0];
    }
    /// Special const access to second element.
    const real& y() const
    {
        BOOST_STATIC_ASSERT(N >= 2);
        return this->elems[1];
    }
    /// Special const access to third element.
    const real& z() const
    {
        BOOST_STATIC_ASSERT(N >= 3);
        return this->elems[2];
    }
    /// Special const access to fourth element.
    const real& w() const
    {
        BOOST_STATIC_ASSERT(N >= 4);
        return this->elems[3];
    }

    /// Assignment operator from an array of values.
    template<typename real2>
    void operator=(const real2* p)
    {
        //std::copy(p, p+N, this->begin());
        for(int i=0; i<N; i++)
            this->elems[i] = (real)p[i];
    }

    /// Assignment from a vector with different dimensions.
    template<int M, typename real2>
    void operator=(const Vec<M,real2>& v)
    {
        //std::copy(v.begin(), v.begin()+(N>M?M:N), this->begin());
        for(int i=0; i<(N>M?M:N); i++)
            this->elems[i] = (real)v(i);
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

    // Access to i-th element.
    // Already in fixed_array
    //real& operator[](int i)
    //{
    //    return this->elems[i];
    //}

    // Access to i-th element.
    // Already in fixed_array
    /// Const access to i-th element.
    //const real& operator[](int i) const
    //{
    //    return this->elems[i];
    //}

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
    /// CHANGE(Jeremie A.): removed it as it confuses some compilers. Use ptr() or data() instead
    //operator const real*() const
    //{
    //    return this->elems;
    //}

    /// Cast into an array of values.
    /// CHANGE(Jeremie A.): removed it as it confuses some compilers. Use ptr() or data() instead
    //operator real*()
    //{
    //    return this->elems;
    //}

    /// Cast into a const array of values.
    const real* ptr() const
    {
        return this->elems;
    }

    /// Cast into an array of values.
    real* ptr()
    {
        return this->elems;
    }

    // LINEAR ALGEBRA

    /// Multiplication by a scalar f.
    template<class real2>
    Vec<N,real> operator*(real2 f) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]*(real)f;
        return r;
    }

    /// On-place multiplication by a scalar f.
    template<class real2>
    void operator*=(real2 f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]*=(real)f;
    }

    /// Division by a scalar f.
    template<class real2>
    Vec<N,real> operator/(real2 f) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i] = this->elems[i]/(real)f;
        return r;
    }

    /// On-place division by a scalar f.
    template<class real2>
    void operator/=(real2 f)
    {
        for (int i=0; i<N; i++)
            this->elems[i]/=(real)f;
    }

    /// Dot product.
    template<class real2>
    real operator*(const Vec<N,real2>& v) const
    {
        real r = this->elems[0]*v[0];
        for (int i=1; i<N; i++)
            r += (real)(this->elems[i]*v[i]);
        return r;
    }

    /// Vector addition.
    template<class real2>
    Vec<N,real> operator+(const Vec<N,real2>& v) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]+(real)v[i];
        return r;
    }

    /// On-place vector addition.
    template<class real2>
    void operator+=(const Vec<N,real2>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]+=(real)v[i];
    }

    /// Vector subtraction.
    template<class real2>
    Vec<N,real> operator-(const Vec<N,real2>& v) const
    {
        Vec<N,real> r;
        for (int i=0; i<N; i++)
            r[i]=this->elems[i]-(real)v[i];
        return r;
    }

    /// On-place vector subtraction.
    template<class real2>
    void operator-=(const Vec<N,real2>& v)
    {
        for (int i=0; i<N; i++)
            this->elems[i]-=(real)v[i];
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

    Vec cross( const Vec& b ) const
    {
        BOOST_STATIC_ASSERT(N == 3);
        return Vec(
                (*this)[1]*b[2] - (*this)[2]*b[1],
                (*this)[2]*b[0] - (*this)[0]*b[2],
                (*this)[0]*b[1] - (*this)[1]*b[0]
                );
    }



};

/// Read from an input stream
template<int N,typename Real>
std::istream& operator >> ( std::istream& in, Vec<N,Real>& v )
{
    for( int i=0; i<N; ++i )
        in>>v[i];
    return in;
}

/// Write to an output stream
template<int N,typename Real>
std::ostream& operator << ( std::ostream& out, const Vec<N,Real>& v )
{
    for( int i=0; i<N-1; ++i )
        out<<v[i]<<" ";
    out<<v[N-1];
    return out;
}

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

typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;

typedef Vec2d Vector2; ///< alias

typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;

typedef Vec3d Vector3; ///< alias

typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;

typedef Vec4d Vector4; ///< alias


} // namespace Common

} // namespace Components

} // namespace Sofa


// Specialization of the std comparison function, to use Vec as std::map key
namespace std
{

// template <>
template<int N, class T>
struct less< Sofa::Components::Common::Vec<N,T> > : public binary_function< Sofa::Components::Common::Vec<N,T>,  Sofa::Components::Common::Vec<N,T>, bool>
{
    bool operator()(const  Sofa::Components::Common::Vec<N,T>& x, const  Sofa::Components::Common::Vec<N,T>& y) const
    {
        //std::cerr<<"specialized std::less, x = "<<x<<", y = "<<y<<std::endl;
        for( unsigned i=0; i<N; ++i )
        {
            if( x[i]<y[i] )
                return true;
            else if( y[i]<x[i] )
                return false;
        }
        return false;
    }
};

} // namespace std

#endif

