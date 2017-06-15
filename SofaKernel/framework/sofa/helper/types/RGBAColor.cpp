/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/types/RGBAColor.h>
#include <sofa/helper/logging/Messaging.h>
#include <sstream>
namespace sofa
{
namespace helper
{
namespace types
{

int hexval(char c)
{
    if (c>='0' && c<='9') return c-'0';
    else if (c>='a' && c<='f') return (c-'a')+10;
    else if (c>='A' && c<='F') return (c-'A')+10;
    else return 0;
}

bool isValidEncoding(const std::string& s)
{
    auto c = s.begin();
    if( *c != '#' )
        return false;

    for( c++ ; c != s.end() ; ++c ){
        if (*c>='0' && *c<='9') {}
        else if (*c>='a' && *c<='f') {}
        else if (*c>='A' && *c<='F') {}
        else return false;
    }
    return true;
}


RGBAColor::RGBAColor() : fixed_array<float, 4>(1,1,1,1)
{

}


RGBAColor::RGBAColor(const fixed_array<float, 4>& c) : fixed_array<float, 4>(c)
{

}



RGBAColor::RGBAColor(const float pr, const float pg, const float pb, const float pa)
{
    r(pr);
    g(pg);
    b(pb);
    a(pa);
}


bool RGBAColor::read(const std::string& str, RGBAColor& color)
{
    if (str.empty())
        return true;

    float r,g,b,a=1.0;
    if (str[0]>='0' && str[0]<='9')
    {
        std::istringstream iss(str);
        iss >> r >> g >> b ;
        if(iss.fail()){
            return false;
        }
        if(!iss.eof()){
            iss >> a;
            if(iss.fail() || !iss.eof()){
                return false;
            }
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
    else {
        return false ;
    }

    color.set(r,g,b,a) ;
    return true ;
}


void RGBAColor::set(float r, float g, float b, float a)
{
    this->elems[0]=r;
    this->elems[1]=g;
    this->elems[2]=b;
    this->elems[3]=a;
}


RGBAColor RGBAColor::fromString(const std::string& c)
{
    RGBAColor color(1.0,1.0,1.0,1.0) ;
    if( !RGBAColor::read(c, color) ){
        msg_info("RGBAColor") << "Unable to scan color from string '" << c << "'" ;
    }
    return color;
}


RGBAColor RGBAColor::fromFloat(const float r, const float g, const float b, const float a)
{
    return RGBAColor(r,g,b,a);
}


RGBAColor RGBAColor::fromVec4(const fixed_array<float, 4>& color)
{
    return RGBAColor(color) ;
}


RGBAColor RGBAColor::fromVec4(const fixed_array<double, 4>& color)
{
    return RGBAColor(color[0], color[1], color[2], color[3]) ;
}

RGBAColor RGBAColor::fromHSVA(float h, float s, float v, float a )
{
    // H [0, 360] S, V and A [0.0, 1.0].
    RGBAColor rgba;

    int i = (int)floor(h/60.0f) % 6;
    float f = h/60.0f - floor(h/60.0f);
    float p = v * (float)(1 - s);
    float q = v * (float)(1 - s * f);
    float t = v * (float)(1 - (1 - f) * s);
    rgba[3]=a;
    switch (i)
    {
    case 0:
        rgba[0]=v; rgba[1]=t; rgba[2]=p;
        break;
    case 1: rgba[0]=q; rgba[1]=v; rgba[2]=p;
        break;
    case 2: rgba[0]=p; rgba[1]=v; rgba[2]=t;
        break;
    case 3: rgba[0]=p; rgba[1]=q; rgba[2]=v;
        break;
    case 4: rgba[0]=t; rgba[1]=p; rgba[2]=v;
        break;
    case 5: rgba[0]=v; rgba[1]=p; rgba[2]=q;
    }

    return rgba;
}

SOFA_HELPER_API std::istream& operator>>(std::istream& i, RGBAColor& t)
{
    std::string s;
    std::getline(i, s);
    if(!RGBAColor::read(s, t)){
        i.setstate(std::ios::failbit) ;
    }

    return i;
}

/// Write to an output stream
SOFA_HELPER_API std::ostream& operator << ( std::ostream& out, const RGBAColor& v )
{
    for( int i=0; i<3; ++i )
        out<<v[i]<<" ";
    out<<v[3];
    return out;
}


} // namespace types
} // namespace helper
} // namespace sofa

