/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Contributions:                                                              *
*     - damien.marchal@univ-lille1.fr                                         *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_DEFAULTTYPE_COLOR_H
#define SOFA_DEFAULTTYPE_COLOR_H
#include <string>

#include <sofa/defaulttype/defaulttype.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace defaulttype
{

/**
 *  \brief encode a 4 RGBA component color as a specialized Vec<4, float> vector.
 */
template<typename T>
class SOFA_DEFAULTTYPE_API TRGBAColor : public Vec<4, T>
{
public:
    static TRGBAColor<T> fromString(const std::string& str) ;
    static TRGBAColor fromDouble(const float r, const float g, const float b, const float a) ;
    static TRGBAColor fromVec4(const Vec4d& color) ;
    static TRGBAColor fromVec4(const Vec4f& color) ;
    static bool read(const std::string& str, TRGBAColor& color) ;

    static TRGBAColor white()  { return TRGBAColor(1.0,1.0,1.0,1.0); }
    static TRGBAColor black()  { return TRGBAColor(0.0,0.0,0.0,1.0); }
    static TRGBAColor red()    { return TRGBAColor(1.0,0.0,0.0,1.0); }
    static TRGBAColor green()  { return TRGBAColor(0.0,1.0,0.0,1.0); }
    static TRGBAColor blue()   { return TRGBAColor(0.0,0.0,1.0,1.0); }
    static TRGBAColor cyan()   { return TRGBAColor(0.0,1.0,1.0,1.0); }
    static TRGBAColor magenta() { return TRGBAColor(1.0,0.0,1.0,1.0); }
    static TRGBAColor yellow()  { return TRGBAColor(1.0,1.0,0.0,1.0); }
    static TRGBAColor gray()    { return TRGBAColor(0.5,0.5,0.5,1.0); }

    using Vec<4,T>::x ;
    using Vec<4,T>::y ;
    using Vec<4,T>::z ;
    using Vec<4,T>::w ;

    float& r(){ return x() ; }
    float& g(){ return y() ; }
    float& b(){ return z() ; }
    float& a(){ return w() ; }
    const float& r() const { return x() ; }
    const float& g() const { return y() ; }
    const float& b() const { return z() ; }
    const float& a() const { return w() ; }

    void r(const float r){ x()=r; }
    void g(const float r){ y()=r; }
    void b(const float r){ z()=r; }
    void a(const float r){ w()=r; }

    friend std::istream& operator>>(std::istream& i, TRGBAColor<float>& t) ;

public:
    TRGBAColor() ;
    TRGBAColor(const  Vec4f&) ;
    TRGBAColor(const float r, const float g, const float b, const float a) ;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_DEFAULTTYPE_COLOR_CPP)
extern template class SOFA_DEFAULTTYPE_API TRGBAColor<float> ;
#endif

typedef TRGBAColor<float> RGBAColor ;

/*
template<>
struct DataTypeInfo< TRGBAColor > : public FixedArrayTypeInfo<Vec4f>
{
    static std::string name() { std::ostringstream o; o << "TRGBAColor" << 4 << "f"; return o.str(); }
};*/

} // namespace defaulttype

} // namespace sofa


#endif

