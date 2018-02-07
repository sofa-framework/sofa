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
#ifndef SOFA_HELPER_RMATH_H
#define SOFA_HELPER_RMATH_H

#include <sofa/helper/helper.h>
#include <cmath>
#include <limits>

namespace sofa
{

namespace helper
{

#ifdef M_PI
#define R_PI M_PI
#else
#define R_PI 3.141592653589793238462
#endif

/*
// Most portable version, but also the slowest
inline int rfloor(double r)
{
  return ((int)(r+1000000))-1000000;
}

inline int rnear(double r)
{
  return ((int)(r+1000000.5))-1000000;
}
*/

/*
// Does not work on gcc 4.x due to more aggressive optimizations
inline int rfloor(double r)
{
  static const double FLOATTOINTCONST=(1.5*(1LL<<(52-16)));
  r+=FLOATTOINTCONST;
  return ((((int*)&r)[0])>>16);
}

inline int rnear(double r)
{
  static const double FLOATTOINTCONST_0_5=(1.5*(1LL<<(52-16)))+0.5;
  r+=FLOATTOINTCONST_0_5;
  return ((((int*)&r)[0])>>16);
}
*/

// Works on gcc 3.x and 4.x
template<class real>
inline int rfloor(real r)
{
    static const double FLOATTOINTCONST=(1.5*(1LL<<(52-16)));
    union
    {
        double d;
        int i;
    } temp;
    temp.d = r+FLOATTOINTCONST;
    return ((temp.i)>>16);
}

template<class real>
inline int rnear(real r)
{
    static const double FLOATTOINTCONST_0_5=(1.5*(1LL<<(52-16)))+0.5;
    union
    {
        double d;
        int i;
    } temp;
    temp.d = r+FLOATTOINTCONST_0_5;
    return ((temp.i)>>16);
}

inline int rceil(double r)
{
    return -rfloor(-r);
}

template<class real>
inline real rabs(real r)
{
    return (r>=0)?r:-r;
}

template<class real>
inline real rmin(real r, real s)
{
    return (r<s)?r:s;
}

template<class real>
inline real rmax(real r, real s)
{
    return (r>s)?r:s;
}

template<class T>
inline T rlerp(const T& a, const T& b, float f)
{
    return a+(b-a)*f;
}

template<class T>
inline T rsqrt(const T& a)
{
    return (T)sqrtf((float)a);
}

inline double rsqrt(const double& a)
{
#if defined(__GNUC__)
    return sqrt(a);
#else
    return (double)sqrtl((long double)a);
#endif
}

inline long double rsqrt(const long double& a)
{
    return sqrtl(a);
}

template<class T>
inline const T SQR(const T& a)
{
    return a*a;
}

template<class T>
inline const T SIGN(const T &a, const T &b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

template<class T>
inline void SWAP(T &a, T &b)
{
    T dum=a;
    a=b;
    b=dum;
}

inline void shft3(double &a, double &b, double &c, const double d)
{
    a=b;
    b=c;
    c=d;
}

template<class T>
inline T round(const T& x)
{
    return (SReal)(x<0?ceil(x-0.5):floor(x+0.5));
}

template<class T>
inline T factorial (T Number)
{
    if (Number<=1) return 1;
    return Number*factorial(Number-1);
}

//inline void printerror( const char* msg )
//{
//    msg_info()<<msg<<std::endl;
//    assert(0);
//}

template<class T>
inline T rclamp(const T& value, const T& low, const T& high)
{
  return value < low ? low : (value > high ? high : value);
}

template<class T>
inline bool isClamped(const T& value, const T& low, const T& high)
{
  return value >= low && value <= high;
}

template<class T>
inline T sign( const T& v )
{
    return v<0 ? (T)-1 : (T)1;
}

template<class T>
inline T sign0( const T& v )
{
    return v<0 ? (T)-1 : ( v>0 ? (T)1 : 0 );
}


// (FF) why do we need a comparison function for integers ? Why not operator == ?
// (MattN) to allow a common code templated for both integers and floating points

/// @internal
template<bool is_integer=false>
struct IsEqual
{
    template<typename T> inline static bool test( T x, T y, T threshold )
    {
       return rabs(x-y) <= threshold;
    }
};

/// @internal specialization for integer types
template<>
struct IsEqual<true>
{
    template<typename T> inline static bool test( T x, T y, T )
    {
       return x==y;
    }
};

/// number comparison
/// rough floating point comparison (threshold)
/// exact integer comparison
template<class T>
inline bool isEqual( T x, T y, T threshold = (std::numeric_limits<T>::epsilon)() )
{
    return IsEqual<std::numeric_limits<T>::is_integer>::test( x, y, threshold );
}



/// @internal
template<bool is_integer=false>
struct IsNull
{
    template<typename T> static bool test( T x, T threshold )
    {
       return rabs(x) <= threshold;
    }
};

/// @internal specialization for integer types
template<>
struct IsNull<true>
{
    template<typename T> static bool test( T x, T )
    {
       return x==0;
    }
};

/// number null test
/// rough floating point test ( <= threshold)
/// exact integer test
template<class T>
inline bool isNull( T x, T threshold = (std::numeric_limits<T>::epsilon)() )
{
    return IsNull<std::numeric_limits<T>::is_integer>::test( x, threshold );
}



inline double rcos(double x){
	return cos(x);
}

inline float rcos(float x){
	return cosf(x);
}

inline double rsin(double x){
	return sin(x);
}

inline float rsin(float x){
	return sinf(x);
}

template<class T>
inline T rcos(const T& a)
{
    return (T)cos((double)a);
}

template<class T>
inline T rsin(const T& a)
{
    return (T)sin((double)a);
}

} // namespace helper

} // namespace sofa

#endif


