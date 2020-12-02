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

#include <sofa/type/stdtype/fixed_array.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>

#define EQUALITY_THRESHOLD 1e-6

namespace sofa::type
{

namespace // anonymous
{
    template<typename real>
    real rabs(const real r)
    {
        if constexpr (std::is_signed<real>())
            return std::abs(r);
        else
            return r;
    }

} // anonymous namespace

//enum NoInit { NOINIT }; ///< use when calling Vec or Mat constructor to skip initialization of values to 0
struct NoInit {};
constexpr NoInit NOINIT;

template < sofa::Size N, typename ValueType=float>
class Vec : public sofa::type::stdtype::fixed_array<ValueType,size_t(N)>
{

    static_assert( N > 0, "" );

public:
    typedef sofa::Size Size;

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to static_size and size() method)
    enum { total_size = N };
    /// Compile-time constant specifying the number of dimensions of space (equivalent to total_size here)
    enum { spatial_dimensions = N };

    /// Default constructor: sets all values to 0.
    Vec()
    {
        this->clear();
    }

    /// Fast constructor: no initialization
    explicit Vec(NoInit)
    {
    }

    /// Specific constructor for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN==1,int>::type = 0>
    explicit Vec(ValueType r1)
    {
        set( r1 );
    }

    /// Specific constructor for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN==1,int>::type = 0>
    void operator=(ValueType r1)
    {
        set( r1 );
    }

    /// Specific constructor for 2-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==2,int>::type = 0>
    Vec(ValueType r1, ValueType r2)
    {
        set( r1, r2 );
    }

    /// Specific constructor for 3-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==3,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3)
    {
        set( r1, r2, r3 );
    }

    /// Specific constructor for 4-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==4,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4)
    {
        set( r1, r2, r3, r4 );
    }

    /// Specific constructor for 5-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==5,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5)
    {
        set( r1, r2, r3, r4, r5 );
    }

    /// Specific constructor for 6-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==6,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6)
    {
        set( r1, r2, r3, r4, r5, r6 );
    }

    /// Specific constructor for 6-elements vectors.
    template<typename R, typename T, Size NN=N, typename std::enable_if<NN==6,int>::type = 0 >
    Vec( const Vec<3,R>& a , const Vec<3,T>& b )
    {
        set( a[0], a[1], a[2], b[0], b[1], b[2] );
    }

    /// Specific constructor for 7-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==7,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7)
    {
        set( r1, r2, r3, r4, r5, r6, r7 );
    }

    /// Specific constructor for 8-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==8,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8)
    {
        set( r1, r2, r3, r4, r5, r6, r7, r8 );
    }

    /// Specific constructor for 9-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==9,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8, ValueType r9)
    {
        set( r1, r2, r3, r4, r5, r6, r7, r8, r9 );
    }

    /// Specific constructor for 12-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==12,int>::type = 0>
    Vec(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8, ValueType r9, ValueType r10, ValueType r11, ValueType r12)
    {
        set( r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 );
    }

    /// Specific set for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN==1,int>::type = 0>
    void set(ValueType r1)
    {
        this->elems[0]=r1;
    }

    /// Specific set for 2-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==2,int>::type = 0>
    void set(ValueType r1, ValueType r2)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
    }

    /// Specific set for 3-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==3,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
    }

    /// Specific set for 4-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==4,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
    }

    /// Specific set for 5-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==5,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
    }

    /// Specific set for 6-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==6,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
    }

    /// Specific constructor for 7-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==7,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
    }

    /// Specific set for 8-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==8,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
    }

    /// Specific set for 9-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==9,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8, ValueType r9)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
        this->elems[8]=r9;
    }

    /// Specific set for 12-elements vectors.
    template<Size NN = N, typename std::enable_if<NN==12,int>::type = 0>
    void set(ValueType r1, ValueType r2, ValueType r3, ValueType r4, ValueType r5, ValueType r6, ValueType r7, ValueType r8, ValueType r9, ValueType r10, ValueType r11, ValueType r12)
    {
        this->elems[0]=r1;
        this->elems[1]=r2;
        this->elems[2]=r3;
        this->elems[3]=r4;
        this->elems[4]=r5;
        this->elems[5]=r6;
        this->elems[6]=r7;
        this->elems[7]=r8;
        this->elems[8]=r9;
        this->elems[9]=r10;
        this->elems[10]=r11;
        this->elems[11]=r12;
    }

    /// Specific set from a different size vector (given default value and ignored outside entries)
    template<Size N2, class real2>
    void set(const Vec<N2,real2>& v, ValueType defaultvalue=0)
    {
        Size maxN = std::min( N, N2 );
        for(Size i=0; i<maxN; i++)
            this->elems[i] = (ValueType)v[i];
        for(Size i=maxN; i<N ; i++)
            this->elems[i] = defaultvalue;
    }


    /// Constructor from an N-1 elements vector and an additional value (added at the end).
    template<Size NN = N, typename std::enable_if<(NN>1),int>::type = 0>
    Vec(const Vec<N-1,ValueType>& v, ValueType r1)
    {
        set( v, r1 );
    }

    Vec(const sofa::type::stdtype::fixed_array<ValueType, N>& p)
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = p[i];
    }

    /// Constructor from a different size vector (null default value and ignoring outside entries)
    template<Size N2, typename real2>
    explicit Vec(const Vec<N2,real2>& v)
    {
        set( v, 0 );
    }

    template<typename real2>
    Vec(const Vec<N, real2>& p)
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = (ValueType)p(i);
    }

    /// Constructor from an array of values.
    template<typename real2>
    explicit Vec(const real2* p)
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = (ValueType)p[i];
    }

    /// Special access to first element.
    template<Size NN = N, typename std::enable_if<(NN>=1),int>::type = 0>
    ValueType& x()
    {
        return this->elems[0];
    }
    /// Special access to second element.
    template<Size NN = N, typename std::enable_if<(NN>=2),int>::type = 0>
    ValueType& y()
    {
        return this->elems[1];
    }
    /// Special access to third element.
    template<Size NN = N, typename std::enable_if<(NN>=3),int>::type = 0>
    ValueType& z()
    {
        return this->elems[2];
    }
    /// Special access to fourth element.
    template<Size NN = N, typename std::enable_if<(NN>=4),int>::type = 0>
    ValueType& w()
    {
        return this->elems[3];
    }

    /// Special const access to first element.
    template<Size NN = N, typename std::enable_if<(NN>=1),int>::type = 0>
    const ValueType& x() const
    {
        return this->elems[0];
    }
    /// Special const access to second element.
    template<Size NN = N, typename std::enable_if<(NN>=2),int>::type = 0>
    const ValueType& y() const
    {
        return this->elems[1];
    }
    /// Special const access to third element.
    template<Size NN = N, typename std::enable_if<(NN>=3),int>::type = 0>
    const ValueType& z() const
    {
        return this->elems[2];
    }
    /// Special const access to fourth element.
    template<Size NN = N, typename std::enable_if<(NN>=4),int>::type = 0>
    const ValueType& w() const
    {
        return this->elems[3];
    }

    /// Assignment operator from an array of values.
    template<typename real2>
    void operator=(const real2* p)
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = (ValueType)p[i];
    }

    /// Assignment from a vector with different dimensions.
    template<Size M, typename real2>
    void operator=(const Vec<M,real2>& v)
    {
        for(Size i=0; i<(N>M?M:N); i++)
            this->elems[i] = (ValueType)v(i);
    }

    /// Sets every element to 0.
    inline void clear()
    {
        this->assign(ValueType());
    }

    /// Sets every element to r.
    inline void fill(ValueType r)
    {
        this->assign(r);
    }

    // Access to i-th element.
    // Already in fixed_array
    //real& operator[](Size i)
    //{
    //    return this->elems[i];
    //}

    // Access to i-th element.
    // Already in fixed_array
    /// Const access to i-th element.
    //const real& operator[](Size i) const
    //{
    //    return this->elems[i];
    //}

    /// Access to i-th element.
    ValueType& operator()(Size i)
    {
        return this->elems[i];
    }

    /// Const access to i-th element.
    const ValueType& operator()(Size i) const
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
    const ValueType* ptr() const
    {
        return this->elems;
    }

    /// Cast into an array of values.
    ValueType* ptr()
    {
        return this->elems;
    }

    // LINEAR ALGEBRA

    // BUG (J.A. 12/31/2010): gcc 4.0 does not support templated
    // operators that are restricted to scalar type using static_assert.
    // So for now we are defining them as templated method, and the
    // operators then simply call them with the right type.

    Vec<N,ValueType> mulscalar(ValueType f) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i] = this->elems[i]*f;
        return r;
    }

    /// Multiplication by a scalar f.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> mulscalar(real2 f) const
    {
        return mulscalar((ValueType)f);
    }

    Vec<N,ValueType> operator*(         float     f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(         double    f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(         int       f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(unsigned int       f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(         long      f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(unsigned long      f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(         long long f) const {  return mulscalar((ValueType)f);  }
    Vec<N,ValueType> operator*(unsigned long long f) const {  return mulscalar((ValueType)f);  }

    /// In-place multiplication by a scalar f.
    void eqmulscalar(ValueType f)
    {
        for (Size i=0; i<N; i++)
            this->elems[i]*=f;
    }

    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    void eqmulscalar(real2 f)
    {
        eqmulscalar((ValueType)f);
    }

    void operator*=(         float     f) {  eqmulscalar((ValueType)f);  }
    void operator*=(         double    f) {  eqmulscalar((ValueType)f);  }
    void operator*=(         int       f) {  eqmulscalar((ValueType)f);  }
    void operator*=(unsigned int       f) {  eqmulscalar((ValueType)f);  }
    void operator*=(         long      f) {  eqmulscalar((ValueType)f);  }
    void operator*=(unsigned long      f) {  eqmulscalar((ValueType)f);  }
    void operator*=(         long long f) {  eqmulscalar((ValueType)f);  }
    void operator*=(unsigned long long f) {  eqmulscalar((ValueType)f);  }

    /// Division by a scalar f.
    Vec<N,ValueType> divscalar(ValueType f) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i] = this->elems[i]/f;
        return r;
    }

    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> divscalar(real2 f) const
    {
        return divscalar((ValueType)f);
    }

    Vec<N,ValueType> operator/(         float     f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(         double    f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(         int       f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(unsigned int       f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(         long      f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(unsigned long      f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(         long long f) const {  return divscalar((ValueType)f);  }
    Vec<N,ValueType> operator/(unsigned long long f) const {  return divscalar((ValueType)f);  }

    /// In-place division by a scalar f.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    void eqdivscalar(real2 f)
    {
        eqdivscalar((ValueType)f);
    }

    void eqdivscalar(ValueType f)
    {
        for (Size i=0; i<N; i++)
            this->elems[i]/=f;
    }

    void operator/=(         float     f) {  eqdivscalar((ValueType)f);  }
    void operator/=(         double    f) {  eqdivscalar((ValueType)f);  }
    void operator/=(         int       f) {  eqdivscalar((ValueType)f);  }
    void operator/=(unsigned int       f) {  eqdivscalar((ValueType)f);  }
    void operator/=(         long      f) {  eqdivscalar((ValueType)f);  }
    void operator/=(unsigned long      f) {  eqdivscalar((ValueType)f);  }
    void operator/=(         long long f) {  eqdivscalar((ValueType)f);  }
    void operator/=(unsigned long long f) {  eqdivscalar((ValueType)f);  }

    /// Dot product.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    ValueType operator*(const Vec<N,real2>& v) const
    {
        ValueType r = (ValueType)(this->elems[0]*v[0]);
        for (Size i=1; i<N; i++)
            r += (ValueType)(this->elems[i]*v[i]);
        return r;
    }

    /// linear product.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> linearProduct(const Vec<N,real2>& v) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]*(ValueType)v[i];
        return r;
    }


    /// linear division.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> linearDivision(const Vec<N,real2>& v) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]/(ValueType)v[i];
        return r;
    }

    /// Vector addition.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> operator+(const Vec<N,real2>& v) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]+(ValueType)v[i];
        return r;
    }

    /// In-place vector addition.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    void operator+=(const Vec<N,real2>& v)
    {
        for (Size i=0; i<N; i++)
            this->elems[i]+=(ValueType)v[i];
    }

    /// Vector subtraction.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    Vec<N,ValueType> operator-(const Vec<N,real2>& v) const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]-(ValueType)v[i];
        return r;
    }

    /// In-place vector subtraction.
    template<class real2, std::enable_if_t<std::is_arithmetic_v<real2>, bool> = true>
    void operator-=(const Vec<N,real2>& v)
    {
        for (Size i=0; i<N; i++)
            this->elems[i]-=(ValueType)v[i];
    }

    /// Vector negation.
    Vec<N,ValueType> operator-() const
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=-this->elems[i];
        return r;
    }

    /// Squared norm.
    ValueType norm2() const
    {
        ValueType r = this->elems[0]*this->elems[0];
        for (Size i=1; i<N; i++)
            r += this->elems[i]*this->elems[i];
        return r;
    }

    /// Euclidean norm.
    ValueType norm() const
    {
        return ValueType(std::sqrt(norm2()));
    }

    /// l-norm of the vector
    /// The type of norm is set by parameter l.
    /// Use l<0 for the infinite norm.
    ValueType lNorm( int l ) const
    {
        if( l==2 ) return norm(); // euclidian norm
        else if( l<0 ) // infinite norm
        {
            ValueType n=0;
            for( Size i=0; i<N; i++ )
            {
                ValueType a = rabs( this->elems[i] );
                if( a>n ) n=a;
            }
            return n;
        }
        else if( l==1 ) // Manhattan norm
        {
            ValueType n=0;
            for( Size i=0; i<N; i++ )
            {
                n += rabs( this->elems[i] );
            }
            return n;
        }
        else if( l==0 ) // counting not null
        {
            ValueType n=0;
            for( Size i=0; i<N; i++ )
                if( this->elems[i] ) n+=1;
            return n;
        }
        else // generic implementation
        {
            ValueType n = 0;
            for( Size i=0; i<N; i++ )
                n += ValueType(pow( rabs( this->elems[i] ), l ));
            return ValueType(pow( n, ValueType(1.0)/(ValueType)l ));
        }
    }


    /// Normalize the vector taking advantage of its already computed norm, equivalent to /=norm
    /// returns false iff the norm is too small
    bool normalizeWithNorm(ValueType norm, ValueType threshold=std::numeric_limits<ValueType>::epsilon())
    {
        if (norm>threshold)
        {
            for (Size i=0; i<N; i++)
                this->elems[i]/=norm;
            return true;
        }
        else
            return false;
    }

    /// Normalize the vector.
    /// returns false iff the norm is too small
    bool normalize(ValueType threshold=std::numeric_limits<ValueType>::epsilon())
    {
        return normalizeWithNorm(norm(),threshold);
    }

    /// Normalize the vector with a failsafe.
    /// If the norm is too small, the vector becomes the failsafe.
    void normalize(Vec<N,ValueType> failsafe, ValueType threshold=std::numeric_limits<ValueType>::epsilon())
    {
        if( !normalize(threshold) ) *this=failsafe;
    }

    /// Return the normalized vector.
    /// @warning 'this' is not normalized.
    Vec<N,ValueType> normalized() const
    {
        Vec<N,ValueType> r(*this);
        r.normalize();
        return r;
    }

    /// return true if norm()==1
    bool isNormalized( ValueType threshold=std::numeric_limits<ValueType>::epsilon()*(ValueType)10 ) const 
    { 
        return rabs( norm2()-(ValueType)1) <= threshold; 
    }

    template<typename R,Size NN = N, typename std::enable_if<(NN==3),int>::type = 0>
    Vec cross( const Vec<3,R>& b ) const
    {
        return Vec(
                (ValueType)((*this)[1]*b[2] - (*this)[2]*b[1]),
                (ValueType)((*this)[2]*b[0] - (*this)[0]*b[2]),
                (ValueType)((*this)[0]*b[1] - (*this)[1]*b[0])
                );
    }


    /// sum of all elements of the vector
    ValueType sum() const
    {
        ValueType sum = ValueType(0.0);
        for (Size i=0; i<N; i++)
            sum += this->elems[i];
        return sum;
    }


    /// @name Tests operators
    /// @{

    bool operator==(const Vec& b) const
    {
        for (Size i=0; i<N; i++)
            if ( fabs( (float)(this->elems[i] - b[i]) ) > EQUALITY_THRESHOLD ) return false;
        return true;
    }

    bool operator!=(const Vec& b) const
    {
        for (Size i=0; i<N; i++)
            if ( fabs( (float)(this->elems[i] - b[i]) ) > EQUALITY_THRESHOLD ) return true;
        return false;
    }

    /// @}
};


/// Same as Vec except the values are not initialized by default
template <sofa::Size N, typename real=float>
class VecNoInit : public Vec<N,real>
{
public:
    VecNoInit()
        : Vec<N,real>(NOINIT)
    {
    }

    /// Assignment operator from an array of values.
    template<typename real2>
    void operator=(const real2* p)
    {
        this->Vec<N,real>::operator=(p);
    }

    /// Assignment from a vector with different dimensions.
    template<sofa::Size M, typename real2>
    void operator=(const Vec<M,real2>& v)
    {
        this->Vec<N,real>::operator=(v);
    }

    /// Scalar vector multiplication operator.
    friend Vec<N,real> operator*(real r, const Vec<N,real>& v)
    {
        return v*r;
    }
};

/// Read from an input stream
template<sofa::Size N,typename Real>
std::istream& operator >> ( std::istream& in, Vec<N,Real>& v )
{
    for(sofa::Size i=0; i<N; ++i )
        in>>v[i];
    return in;
}

/// Write to an output stream
template<sofa::Size N,typename Real>
std::ostream& operator << ( std::ostream& out, const Vec<N,Real>& v )
{
    for(sofa::Size i=0; i<N-1; ++i )
        out<<v[i]<<" ";
    out<<v[N-1];
    return out;
}

/// Cross product for 3-elements vectors.
template<typename real1, typename real2 >
inline Vec<3,real1> cross(const Vec<3,real1>& a, const Vec<3,real2>& b)
{
    return Vec<3,real1>((real1)(a.y()*b.z() - a.z()*b.y()),
            (real1)(a.z()*b.x() - a.x()*b.z()),
            (real1)(a.x()*b.y() - a.y()*b.x()));
}

/// Cross product for 2-elements vectors.
template <typename real1, typename real2>
real1 cross(const type::Vec<2,real1>& a, const type::Vec<2,real2>& b )
{
    return (real1)(a[0]*b[1] - a[1]*b[0]);
}

/// Dot product (alias for operator*)
template<sofa::Size N,typename real>
inline real dot(const Vec<N,real>& a, const Vec<N,real>& b)
{
    return a*b;
}

///// multiplication with a scalar \returns a*V
template <sofa::Size N, typename real>
Vec<N,real> operator*(const double& a, const Vec<N,real>& V)
{
    return V * a;
}

///// multiplication with a scalar \returns a*V
template <sofa::Size N, typename real>
Vec<N,real> operator*(const float& a, const Vec<N,real>& V)
{
    return V * a;
}

typedef Vec<1,float> Vec1f;
typedef Vec<1,double> Vec1d;
typedef Vec<1,int> Vec1i;
typedef Vec<1,unsigned> Vec1u;
typedef Vec<1,SReal> Vec1;

typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;
typedef Vec<2,int> Vec2i;
typedef Vec<2,unsigned> Vec2u;
typedef Vec<2,SReal> Vec2;

typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;
typedef Vec<3,int> Vec3i;
typedef Vec<3,unsigned> Vec3u;
typedef Vec<3,SReal> Vec3;

typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;
typedef Vec<4,int> Vec4i;
typedef Vec<4,unsigned> Vec4u;
typedef Vec<4,SReal> Vec4;

typedef Vec<6,float> Vec6f;
typedef Vec<6,double> Vec6d;
typedef Vec<6,int> Vec6i;
typedef Vec<6,unsigned> Vec6u;
typedef Vec<6,SReal> Vec6;

typedef Vec1d Vector1; ///< alias
typedef Vec2d Vector2; ///< alias
typedef Vec3d Vector3; ///< alias
typedef Vec4d Vector4; ///< alias
typedef Vec6d Vector6; ///< alias

} // namespace sofa::type

// Specialization of the std comparison function, to use Vec as std::map key
namespace std
{

// template <>
template<sofa::Size N, class T>
struct less< sofa::type::Vec<N,T> >
{
    bool operator()(const  sofa::type::Vec<N,T>& x, const  sofa::type::Vec<N,T>& y) const
    {
        //msg_info()<<"specialized std::less, x = "<<x<<", y = "<<y<<std::endl;
        for(sofa::Size i=0; i<N; ++i )
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
