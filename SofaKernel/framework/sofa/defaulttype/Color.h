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

extern int hexval(char c) ;
extern bool isValidEncoding(const std::string& s) ;
//extern SOFA_DEFAULTTYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t) ;

/**
 *  \brief encode a 4 RGBA component color as a specialized Vec<4, float> vector.
 */
class SOFA_DEFAULTTYPE_API RGBAColor : public Vec<4, float>
{
public:
    RGBAColor()
    {

    }


    RGBAColor(const Vec4f& c) : Vec4f(c)
    {

    }


    RGBAColor(const float pr, const float pg, const float pb, const float pa)
    {
        r(pr);
        g(pg);
        b(pb);
        a(pa);
    }


    static bool read(const std::string& str, RGBAColor& color)
    {
        if (str.empty())
            return true;

        float r,g,b,a=1.0;
        if (str[0]>='0' && str[0]<='9')
        {
            std::istringstream iss(str);
            iss >> r >> g >> b ;
            if(iss.fail())
                return false;
            if(!iss.eof()){
                iss >> a;
                if(iss.fail() || !iss.eof())
                    return false;
            }
        }
        else if (str[0]=='#' && str.length()>=7)
        {
            if(!isValidEncoding(str))
                return false;

            r = (hexval(str[1])*16+hexval(str[2]))/255.0f;
            g = (hexval(str[3])*16+hexval(str[4]))/255.0f;
            b = (hexval(str[5])*16+hexval(str[6]))/255.0f;
            if (str.length()>=9)
                a = (hexval(str[7])*16+hexval(str[8]))/255.0f;
            if (str.length()>9)
                return false;
        }
        else if (str[0]=='#' && str.length()>=4)
        {
            if(!isValidEncoding(str))
                return false;

            r = (hexval(str[1])*17)/255.0f;
            g = (hexval(str[2])*17)/255.0f;
            b = (hexval(str[3])*17)/255.0f;
            if (str.length()>=5)
                a = (hexval(str[4])*17)/255.0f;
            if (str.length()>5)
                return false;
        }
        /// If you add more colors... please also add them in the test file.
        else if (str == "white")    { r = 1.0f; g = 1.0f; b = 1.0f; }
        else if (str == "black")    { r = 0.0f; g = 0.0f; b = 0.0f; }
        else if (str == "red")      { r = 1.0f; g = 0.0f; b = 0.0f; }
        else if (str == "green")    { r = 0.0f; g = 1.0f; b = 0.0f; }
        else if (str == "blue")     { r = 0.0f; g = 0.0f; b = 1.0f; }
        else if (str == "cyan")     { r = 0.0f; g = 1.0f; b = 1.0f; }
        else if (str == "magenta")  { r = 1.0f; g = 0.0f; b = 1.0f; }
        else if (str == "yellow")   { r = 1.0f; g = 1.0f; b = 0.0f; }
        else if (str == "gray")     { r = 0.5f; g = 0.5f; b = 0.5f; }
        else
        {
            return false ;
        }

        color.set(r,g,b,a) ;
        return true ;
    }


    static RGBAColor fromString(const std::string& c)
    {
        RGBAColor color(1.0,1.0,1.0,1.0) ;
        if( !RGBAColor::read(c, color) ){
            msg_info("RGBAColor") << "Unable to scan color from string '" << c << "'" ;
        }
        return color;
    }


    static RGBAColor fromDouble(const float r, const float g, const float b, const float a)
    {
        return RGBAColor(r,g,b,a);
    }


    static RGBAColor fromVec4(const Vec4d& color)
    {
        return RGBAColor(color) ;
    }


    static RGBAColor fromVec4(const Vec4f& color)
    {
        return RGBAColor(color.x(), color.y(), color.z(), color.w()) ;
    }


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
};



} // namespace defaulttype

} // namespace sofa


#endif

