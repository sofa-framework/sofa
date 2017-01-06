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

#include <sofa/helper/system/config.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace defaulttype
{


/**
 *  \brief encode a 4 RGBA component color as a specialized Vec<4, double> vector.
 */
class SOFA_DEFAULTTYPE_API RGBAColor : public Vec<4, double>
{
public:
    static RGBAColor fromString(const std::string& str) ;
    static RGBAColor fromDouble(const double r, const double g, const double b, const double a) ;
    static RGBAColor fromVec4(const Vec4d color) ;
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

    double& r(){ return x() ; }
    double& g(){ return y() ; }
    double& b(){ return z() ; }
    double& a(){ return w() ; }
    void r(const double r){ x()=r; }
    void g(const double r){ y()=r; }
    void b(const double r){ z()=r; }
    void a(const double r){ w()=r; }

    friend std::istream& operator>>(std::istream& i, RGBAColor& t) ;

public:
    RGBAColor() ;
    RGBAColor(const  Vec4d&) ;
    RGBAColor(const double r, const double g, const double b, const double a) ;

};


template<>
struct DataTypeInfo< RGBAColor > : public FixedArrayTypeInfo<sofa::defaulttype::Vec<4, double> >
{
    static std::string name() { std::ostringstream o; o << "RGBAColor" << 4 << "d"; return o.str(); }
};

} // namespace defaulttype

} // namespace sofa


#endif

