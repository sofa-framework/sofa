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
class SOFA_DEFAULTTYPE_API RGBAColor : public Vec<4, float>
{
public:
    static RGBAColor fromString(const std::string& str) ;
    static RGBAColor fromDouble(const float r, const float g, const float b, const float a) ;
    static RGBAColor fromVec4(const Vec4d& color) ;
    static RGBAColor fromVec4(const Vec4f& color) ;
    static bool read(const std::string& str, RGBAColor& color) ;

    static RGBAColor white()  { return RGBAColor(1.0,1.0,1.0,1.0); }
    static RGBAColor black()  { return RGBAColor(0.0,0.0,0.0,1.0); }
    static RGBAColor red()    { return RGBAColor(1.0,0.0,0.0,1.0); }
    static RGBAColor green()  { return RGBAColor(0.0,1.0,0.0,1.0); }
    static RGBAColor blue()   { return RGBAColor(0.0,0.0,1.0,1.0); }
    static RGBAColor cyan()   { return RGBAColor(0.0,1.0,1.0,1.0); }
    static RGBAColor magenta() { return RGBAColor(1.0,0.0,1.0,1.0); }
    static RGBAColor yellow()  { return RGBAColor(1.0,1.0,0.0,1.0); }
    static RGBAColor gray()    { return RGBAColor(0.5,0.5,0.5,1.0); }

    using Vec<4,float>::x ;
    using Vec<4,float>::y ;
    using Vec<4,float>::z ;
    using Vec<4,float>::w ;

    inline float& r(){ return x() ; }
    inline float& g(){ return y() ; }
    inline float& b(){ return z() ; }
    inline float& a(){ return w() ; }
    inline const float& r() const { return x() ; }
    inline const float& g() const { return y() ; }
    inline const float& b() const { return z() ; }
    inline const float& a() const { return w() ; }

    inline void r(const float r){ x()=r; }
    inline void g(const float r){ y()=r; }
    inline void b(const float r){ z()=r; }
    inline void a(const float r){ w()=r; }

    friend SOFA_DEFAULTTYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t) ;

public:
    RGBAColor() ;
    RGBAColor(const  Vec4f&) ;
    RGBAColor(const float r, const float g, const float b, const float a) ;

};

} // namespace defaulttype

} // namespace sofa


#endif

