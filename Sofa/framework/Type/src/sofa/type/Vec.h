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

#include <sofa/type/fixed_array.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>
#include <sofa/type/fwd.h>
#include <cmath>
#include <array>

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

template < sofa::Size N, typename ValueType>
class Vec
{
    static_assert( N > 0, "" );

public:
    using ArrayType = std::array<ValueType, N>;
    ArrayType elems{};

    typedef sofa::Size                          Size;
    typedef ValueType                           value_type;
    typedef typename ArrayType::iterator        iterator;
    typedef typename ArrayType::const_iterator  const_iterator;
    typedef typename ArrayType::reference       reference;
    typedef typename ArrayType::const_reference const_reference;
    typedef sofa::Size       size_type;
    typedef std::ptrdiff_t   difference_type;

    static constexpr sofa::Size static_size = N;
    static constexpr sofa::Size size() { return static_size; }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to static_size and size() method)
    static constexpr Size total_size = N;
    /// Compile-time constant specifying the number of dimensions of space (equivalent to total_size here)
    static constexpr Size spatial_dimensions = N;

    /// Default constructor: sets all values to 0.
    constexpr Vec() = default;

    /// Fast constructor: no initialization
    explicit constexpr Vec(NoInit)
    {
    }

    /// Specific constructor for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN == 1, int>::type = 0>
    explicit constexpr Vec(const ValueType r1) noexcept
    {
        this->set(r1);
    }

    template<typename... ArgsT,
        typename = std::enable_if_t< (std::is_convertible_v<ArgsT, ValueType> && ...) >,
        typename = std::enable_if_t< (sizeof...(ArgsT) == N && sizeof...(ArgsT) > 1) >
    >
    constexpr Vec(ArgsT&&... r) noexcept
        : elems{ static_cast<value_type>(std::forward< ArgsT >(r))... }
    {}

    /// Specific constructor for 6-elements vectors, taking two 3-elements vectors
    template<typename R, typename T, Size NN = N, typename std::enable_if<NN == 6, int>::type = 0 >
    Vec(const Vec<3, R>& a, const Vec<3, T>& b)
    {
        set(a[0], a[1], a[2], b[0], b[1], b[2]);
    }

    /// Specific set function for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN == 1, int>::type = 0>
    constexpr void set(const ValueType r1) noexcept
    {
        this->elems[0] = r1;
    }

    template<typename... ArgsT,
        typename = std::enable_if_t< (std::is_convertible_v<ArgsT, ValueType> && ...) >,
        typename = std::enable_if_t< (sizeof...(ArgsT) == N && sizeof...(ArgsT) > 1) >
    >
    constexpr void set(const ArgsT... r) noexcept
    {
        std::size_t i = 0;
        ((this->elems[i++] = r), ...);
    }

    /// Specific set from a different size vector (given default value and ignored outside entries)
    template<Size N2, class real2>
    constexpr void set(const Vec<N2,real2>& v, ValueType defaultvalue=0) noexcept
    {
        constexpr Size maxN = std::min( N, N2 );
        for(Size i=0; i<maxN; i++)
            this->elems[i] = static_cast<ValueType>(v[i]);
        for(Size i=maxN; i<N ; i++)
            this->elems[i] = defaultvalue;
    }


    /// Constructor from an N-1 elements vector and an additional value (added at the end).
    template<Size NN = N, typename std::enable_if<(NN>1),int>::type = 0>
    constexpr Vec(const Vec<N-1,ValueType>& v, ValueType r1) noexcept
    {
        set( v, r1 );
    }

    constexpr Vec(const sofa::type::fixed_array<ValueType, N>& p) noexcept
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = p[i];
    }

    /// Constructor from a different size vector (null default value and ignoring outside entries)
    template<Size N2, typename real2>
    explicit constexpr Vec(const Vec<N2,real2>& v) noexcept
    {
        set( v, 0 );
    }

    template<typename real2>
    constexpr Vec(const Vec<N, real2>& p) noexcept
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = static_cast<ValueType>(p(i));
    }

    /// Constructor from an array of values.
    template<typename real2>
    explicit constexpr Vec(const real2* p) noexcept
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = static_cast<ValueType>(p[i]);
    }

    /// Special access to first element.
    template<Size NN = N, typename std::enable_if<(NN>=1),int>::type = 0>
    constexpr ValueType& x() noexcept
    {
        return this->elems[0];
    }
    /// Special access to second element.
    template<Size NN = N, typename std::enable_if<(NN>=2),int>::type = 0>
    constexpr ValueType& y() noexcept
    {
        return this->elems[1];
    }
    /// Special access to third element.
    template<Size NN = N, typename std::enable_if<(NN>=3),int>::type = 0>
    constexpr ValueType& z() noexcept
    {
        return this->elems[2];
    }
    /// Special access to fourth element.
    template<Size NN = N, typename std::enable_if<(NN>=4),int>::type = 0>
    constexpr ValueType& w() noexcept
    {
        return this->elems[3];
    }

    /// Special const access to first element.
    template<Size NN = N, typename std::enable_if<(NN>=1),int>::type = 0>
    constexpr const ValueType& x() const noexcept
    {
        return this->elems[0];
    }
    /// Special const access to second element.
    template<Size NN = N, typename std::enable_if<(NN>=2),int>::type = 0>
    constexpr const ValueType& y() const noexcept
    {
        return this->elems[1];
    }
    /// Special const access to third element.
    template<Size NN = N, typename std::enable_if<(NN>=3),int>::type = 0>
    constexpr const ValueType& z() const noexcept
    {
        return this->elems[2];
    }
    /// Special const access to fourth element.
    template<Size NN = N, typename std::enable_if<(NN>=4),int>::type = 0>
    constexpr const ValueType& w() const noexcept
    {
        return this->elems[3];
    }

    /// Specific Assignment operator for 1-element vectors.
    template<Size NN = N, typename std::enable_if<NN == 1, int>::type = 0>
    constexpr void operator=(const ValueType r1) noexcept
    {
        set(r1);
    }

    /// Assignment operator from an array of values.
    template<typename real2>
    constexpr void operator=(const real2* p) noexcept
    {
        for(Size i=0; i<N; i++)
            this->elems[i] = (ValueType)p[i];
    }

    /// Assignment from a vector with different dimensions.
    template<Size M, typename real2>
    constexpr void operator=(const Vec<M,real2>& v) noexcept
    {
        for(Size i=0; i<(N>M?M:N); i++)
            this->elems[i] = (ValueType)v(i);
    }

    // assign one value to all elements
    constexpr void assign(const ValueType& value) noexcept
    {
        for (size_type i = 0; i < N; i++)
            elems[i] = value;
    }

    /// Sets every element to 0.
    constexpr void clear() noexcept
    {
        this->assign(ValueType());
    }

    /// Sets every element to r.
    constexpr void fill(ValueType r) noexcept
    {
        this->assign(r);
    }

    /// Access to i-th element.
    constexpr ValueType& operator()(Size i) noexcept
    {
        return this->elems[i];
    }

    /// Const access to i-th element.
    constexpr const ValueType& operator()(Size i) const noexcept
    {
        return this->elems[i];
    }

    /// Cast into a const array of values.
    constexpr const ValueType* ptr() const noexcept
    {
        return this->elems.data();
    }

    /// Cast into an array of values.
    constexpr ValueType* ptr() noexcept
    {
        return this->elems.data();
    }

    template <Size N2, std::enable_if_t<(N2 < N), bool> = true>
    constexpr void getsub(const Size i, Vec<N2, ValueType>& m) const noexcept
    {
        for (Size j = 0; j < N2; j++)
        {
            m[j] = this->elems[j + i];
        }
    }

    constexpr void getsub(const Size i, ValueType& m) const noexcept
    {
        m = this->elems[i];
    }

    // LINEAR ALGEBRA
    constexpr Vec<N,ValueType> mulscalar(const ValueType f) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i] = this->elems[i]*f;
        return r;
    }

    /// Multiplication by a scalar f.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> mulscalar(const real2 f) const noexcept
    {
        return mulscalar(static_cast<ValueType>(f));
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> operator*(const real2 f) const noexcept
    {
        return mulscalar(static_cast<ValueType>(f));
    }

    /// In-place multiplication by a scalar f.
    constexpr void eqmulscalar(const ValueType f) noexcept
    {
        for (Size i=0; i<N; i++)
            this->elems[i]*=f;
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void eqmulscalar(const real2 f) noexcept
    {
        eqmulscalar(static_cast<ValueType>(f));
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void operator*=(const real2 f) noexcept
    {
        eqmulscalar(static_cast<ValueType>(f));
    }

    /// Division by a scalar f.
    constexpr Vec<N,ValueType> divscalar(const ValueType f) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i] = this->elems[i]/f;
        return r;
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> divscalar(const real2 f) const noexcept
    {
        return divscalar(static_cast<ValueType>(f));
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N, ValueType> operator/(const real2 f) const noexcept
    {
        return divscalar(static_cast<ValueType>(f));
    }

    /// In-place division by a scalar f.
    constexpr void eqdivscalar(const ValueType f) noexcept
    {
        for (Size i = 0; i < N; i++)
            this->elems[i] /= f;
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void eqdivscalar(const real2 f) noexcept
    {
        eqdivscalar(static_cast<ValueType>(f));
    }

    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void operator/=(const real2 f) noexcept
    {
        return eqdivscalar(static_cast<ValueType>(f));
    }

    /// Dot product.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr ValueType operator*(const Vec<N,real2>& v) const noexcept
    {
        ValueType r = static_cast<ValueType>(this->elems[0]*v[0]);
        for (Size i=1; i<N; i++)
            r += static_cast<ValueType>(this->elems[i]*v[i]);
        return r;
    }

    /// linear product.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> linearProduct(const Vec<N,real2>& v) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]* static_cast<ValueType>(v[i]);
        return r;
    }


    /// linear division.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> linearDivision(const Vec<N,real2>& v) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]/ static_cast<ValueType>(v[i]);
        return r;
    }

    /// Vector addition.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> operator+(const Vec<N,real2>& v) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i] + static_cast<ValueType>(v[i]);
        return r;
    }

    /// In-place vector addition.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void operator+=(const Vec<N,real2>& v) noexcept
    {
        for (Size i=0; i<N; i++)
            this->elems[i] += static_cast<ValueType>(v[i]);
    }

    /// Vector subtraction.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr Vec<N,ValueType> operator-(const Vec<N,real2>& v) const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=this->elems[i]-static_cast<ValueType>(v[i]);
        return r;
    }

    /// In-place vector subtraction.
    template<class real2, std::enable_if_t<std::is_convertible_v<real2, ValueType>, bool> = true>
    constexpr void operator-=(const Vec<N,real2>& v) noexcept
    {
        for (Size i=0; i<N; i++)
            this->elems[i] -= static_cast<ValueType>(v[i]);
    }

    /// Vector negation.
    template <typename T = ValueType, std::enable_if_t< !std::is_unsigned_v<T>, int > = 0 >
    constexpr Vec<N, ValueType> operator-() const noexcept
    {
        Vec<N,ValueType> r(NOINIT);
        for (Size i=0; i<N; i++)
            r[i]=-this->elems[i];
        return r;
    }

    /// Squared norm.
    constexpr ValueType norm2() const noexcept
    {
        ValueType r = this->elems[0]*this->elems[0];
        for (Size i=1; i<N; i++)
            r += this->elems[i]*this->elems[i];
        return r;
    }

    /// Euclidean norm.
    ValueType norm() const noexcept
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
                const ValueType a = rabs( this->elems[i] );
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
                n += static_cast<ValueType>((pow( rabs( this->elems[i] ), l )));
            return static_cast<ValueType>(pow( n, static_cast<ValueType>(1.0)/ static_cast<ValueType>(l) ));
        }
    }


    /// Normalize the vector taking advantage of its already computed norm, equivalent to /=norm
    /// returns false iff the norm is too small
    constexpr bool normalizeWithNorm(ValueType norm, ValueType threshold=std::numeric_limits<ValueType>::epsilon()) noexcept
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
    bool normalize(ValueType threshold=std::numeric_limits<ValueType>::epsilon()) noexcept
    {
        return normalizeWithNorm(norm(),threshold);
    }

    /// Normalize the vector with a failsafe.
    /// If the norm is too small, the vector becomes the failsafe.
    void normalize(Vec<N,ValueType> failsafe, ValueType threshold=std::numeric_limits<ValueType>::epsilon()) noexcept
    {
        if( !normalize(threshold) ) *this=failsafe;
    }

    /// Return the normalized vector.
    /// @warning 'this' is not normalized.
    Vec<N,ValueType> normalized() const noexcept
    {
        Vec<N,ValueType> r(*this);
        r.normalize();
        return r;
    }

    /// return true if norm()==1
    bool isNormalized( ValueType threshold=std::numeric_limits<ValueType>::epsilon()*(ValueType)10 ) const
    {
        return rabs( norm2() - static_cast<ValueType>(1) ) <= threshold;
    }

    template<typename R,Size NN = N, typename std::enable_if<(NN==3),int>::type = 0>
    constexpr Vec cross( const Vec<3,R>& b ) const noexcept
    {
        return Vec(
                (ValueType)((*this)[1]*b[2] - (*this)[2]*b[1]),
                (ValueType)((*this)[2]*b[0] - (*this)[0]*b[2]),
                (ValueType)((*this)[0]*b[1] - (*this)[1]*b[0])
                );
    }


    /// sum of all elements of the vector
    constexpr ValueType sum() const noexcept
    {
        ValueType sum = ValueType(0.0);
        for (Size i=0; i<N; i++)
            sum += this->elems[i];
        return sum;
    }


    /// @name Tests operators
    /// @{

    constexpr bool operator==(const Vec& b) const noexcept
    {
        for (Size i=0; i<N; i++)
            if ( fabs( (float)(this->elems[i] - b[i]) ) > EQUALITY_THRESHOLD ) return false;
        return true;
    }

    constexpr bool operator!=(const Vec& b) const noexcept
    {
        for (Size i=0; i<N; i++)
            if ( fabs( (float)(this->elems[i] - b[i]) ) > EQUALITY_THRESHOLD ) return true;
        return false;
    }


    // operator[]
    constexpr reference operator[](size_type i)
    {
        assert(i < N && "index in Vec must be smaller than size");
        return elems[i];
    }
    constexpr const_reference operator[](size_type i) const
    {
        assert(i < N && "index in Vec must be smaller than size");
        return elems[i];
    }

    // direct access to data
    constexpr const ValueType* data() const noexcept
    {
        return elems.data();
    }

    constexpr iterator begin() noexcept
    {
        return elems.begin();
    }
    constexpr const_iterator begin() const noexcept
    {
        return elems.begin();
    }

    constexpr iterator end() noexcept
    {
        return elems.end();
    }
    constexpr const_iterator end() const noexcept
    {
        return elems.end();
    }

    constexpr reference front()
    {
        return elems[0];
    }
    constexpr const_reference front() const
    {
        return elems[0];
    }
    constexpr reference back()
    {
        return elems[N - 1];
    }
    constexpr const_reference back() const
    {
        return elems[N - 1];
    }

    /// @}
};




/// Same as Vec except the values are not initialized by default
template <sofa::Size N, typename real>
class VecNoInit : public Vec<N,real>
{
public:
    constexpr VecNoInit() noexcept
        : Vec<N,real>(NOINIT)
    {}

    constexpr VecNoInit(const Vec<N,real>& v) noexcept
        : Vec<N,real>(v)
    {}

    constexpr VecNoInit(Vec<N,real>&& v) noexcept
        : Vec<N,real>(v)
    {}

    using Vec<N,real>::Vec;

    using Vec<N,real>::operator=; // make every = from Vec available

    /// Scalar vector multiplication operator.
    friend constexpr Vec<N,real> operator*(real r, const Vec<N,real>& v) noexcept
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
constexpr Vec<3,real1> cross(const Vec<3,real1>& a, const Vec<3,real2>& b) noexcept
{
    return Vec<3,real1>((real1)(a.y()*b.z() - a.z()*b.y()),
            (real1)(a.z()*b.x() - a.x()*b.z()),
            (real1)(a.x()*b.y() - a.y()*b.x()));
}

/// Cross product for 2-elements vectors.
template <typename real1, typename real2>
constexpr real1 cross(const type::Vec<2,real1>& a, const type::Vec<2,real2>& b ) noexcept
{
    return (real1)(a[0]*b[1] - a[1]*b[0]);
}

/// Dot product (alias for operator*)
template<sofa::Size N,typename real>
constexpr real dot(const Vec<N,real>& a, const Vec<N,real>& b) noexcept
{
    return a*b;
}

///// multiplication with a scalar \returns a*V
template <sofa::Size N, typename real>
constexpr Vec<N,real> operator*(const double& a, const Vec<N,real>& V) noexcept
{
    return V * a;
}

///// multiplication with a scalar \returns a*V
template <sofa::Size N, typename real>
constexpr Vec<N,real> operator*(const float& a, const Vec<N,real>& V) noexcept
{
    return V * a;
}

/// Checks if v1 is lexicographically less than v2. Similar to std::lexicographical_compare
template<typename T, sofa::Size N>
constexpr bool operator<(const Vec<N, T>& v1, const Vec<N, T>& v2) noexcept
{
    for (sofa::Size i = 0; i < N; i++)
    {
        if (v1[i] < v2[i])
            return true;
        if (v2[i] < v1[i])
            return false;
    }
    return false;
}

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
