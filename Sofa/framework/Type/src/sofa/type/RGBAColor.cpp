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
#include <sofa/type/RGBAColor.h>

#include <sstream>
#include <locale>
#include <map>

#include <sofa/type/fixed_array.h>
#include <sofa/type/fixed_array_algorithms.h>
#include <sofa/type/Vec.h>

using namespace sofa::type::pairwise;

namespace // anonymous
{

    template<class T>
    T rclamp(const T& value, const T& low, const T& high)
    {
        return value < low ? low : (value > high ? high : value);
    }

} // anonymous namespace

namespace sofa::type
{

static bool ishexsymbol(const char c)
{
    return (c>='0' && c<='9') || (c>='a' && c<='f') || (c>='A' && c<='F') ;
}

static int hexval(const char c)
{
    if (c>='0' && c<='9') return c-'0';
    if (c>='a' && c<='f') return (c-'a')+10;
    if (c>='A' && c<='F') return (c-'A')+10;
    return 0;
}


static void extractValidatedHexaString(std::istream& in, std::string& s)
{
    s.reserve(9);
    char c = in.get();

    if(c!='#')
    {
        in.setstate(std::ios_base::failbit) ;
        return;
    }

    s.push_back(c);
    while(in.get(c)){
        if( !ishexsymbol(c) )
            return;

        s.push_back(c) ;
        if(s.size()>9){
            in.setstate(std::ios_base::failbit) ;
            return ;
        }
    }
    /// we need to reset the failbit because it is set by the get function
    /// on the last character.
    in.clear(in.rdstate() & ~std::ios_base::failbit) ;
}


bool RGBAColor::read(const std::string& str, RGBAColor& color)
{
    std::stringstream s(str);
    s >> color ;
    if(s.fail() || !s.eof())
        return false;
    return true ;
}


void RGBAColor::set(const float r, const float g, const float b, const float a)
{
    this->m_components[0] = r;
    this->m_components[1] = g;
    this->m_components[2] = b;
    this->m_components[3] = a;
}


RGBAColor RGBAColor::fromString(const std::string& str)
{
    RGBAColor color(1.0,1.0,1.0,1.0) ;
    if( !RGBAColor::read(str, color) )
    {
        throw std::invalid_argument("Unable to scan color from string '" + str + "'");
    }
    return color;
}


RGBAColor RGBAColor::fromFloat(const float r, const float g, const float b, const float a)
{
    return RGBAColor(r,g,b,a);
}


RGBAColor RGBAColor::fromStdArray(const std::array<float, 4>& color)
{
    return RGBAColor(color) ;
}


RGBAColor RGBAColor::fromStdArray(const std::array<double, 4>& color)
{
    return RGBAColor(float(color[0]), float(color[1]), float(color[2]), float(color[3]));
}


RGBAColor RGBAColor::fromHSVA(const float h, const float s, const float v, const float a )
{
    // H [0, 360] S, V and A [0.0, 1.0].
    RGBAColor rgba;

    const int i = (int)floor(h/60.0f) % 6;
    const float f = h/60.0f - floor(h/60.0f);
    const float p = v * (float)(1 - s);
    const float q = v * (float)(1 - s * f);
    const float t = v * (float)(1 - (1 - f) * s);
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

/// This function remove the leading space in the stream.
static std::istream& trimInitialSpaces(std::istream& in)
{
    char first=in.peek();
    while(!in.eof() && !in.fail() && std::isspace(first, std::locale()))
    {
        in.get();
        first=in.peek();
    }
    return in;
}

const std::map<std::string, RGBAColor> stringToColorMap {
    {"white", g_white},
    {"black", g_black},
    {"red", g_red},
    {"green", g_green},
    {"blue", g_blue},
    {"cyan", g_cyan},
    {"magenta", g_magenta},
    {"yellow", g_yellow},
    {"gray", g_gray},
    {"darkgray", g_darkgray},
    {"lightgray", g_lightgray},
    {"orange", g_orange},
    {"purple", g_purple},
    {"pink", g_pink},
    {"brown", g_brown},
    {"lime", g_lime},
    {"teal", g_teal},
    {"navy", g_navy},
    {"olive", g_olive},
    {"maroon", g_maroon},
    {"silver", g_silver},
    {"gold", g_gold}
};

SOFA_TYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t)
{
    float r=0.0,g=0.0, b=0.0, a=1.0;

    trimInitialSpaces(i) ;

    /// Let's remove the initial spaces.
    if( i.eof() || i.fail() )
        return i;

    const char first = i.peek() ;
    if (std::isdigit(first, std::locale()))
    {
        i >> r >> g >> b ;
        if(!i.eof()){
            i >> a;
        }
    }
    else if (first=='#')
    {
        std::string str;
        extractValidatedHexaString(i, str) ;

        if(i.fail()){
            return i;
        }

        if(str.length()>=7){
            r = (hexval(str[1])*16+hexval(str[2]))/255.0f;
            g = (hexval(str[3])*16+hexval(str[4]))/255.0f;
            b = (hexval(str[5])*16+hexval(str[6]))/255.0f;

            if (str.length()>8)
                a = (hexval(str[7])*16+hexval(str[8]))/255.0f;
        }else if (str.length()>=4){
            r = (hexval(str[1])*17)/255.0f;
            g = (hexval(str[2])*17)/255.0f;
            b = (hexval(str[3])*17)/255.0f;

            if (str.length()>4)
                a = (hexval(str[4])*17)/255.0f;
        }else{
            /// If we cannot parse the field we returns that with the fail bit.
            i.setstate(std::ios_base::failbit) ;
            return i;
        }
    } else {
        std::string str;
        /// Search for the first word, it is not needed to read more char than size("magenta"
        std::getline(i, str, ' ');

        /// if end of line is returned before encountering ' ' or 7... is it fine
        /// so we can clear the failure bitset.
        if (const auto it = stringToColorMap.find(str);
            it != stringToColorMap.end())
        {
            r = it->second.r();
            g = it->second.g();
            b = it->second.b();
            a = it->second.a();
        }
        else {
            /// If we cannot parse the field we returns that with the fail bit.
            i.setstate(std::ios_base::failbit) ;
            return i;
        }
    }

    t.set(r,g,b,a) ;
    return i;
}


/// Write to an output stream
SOFA_TYPE_API std::ostream& operator << ( std::ostream& out, const RGBAColor& t )
{
    for( int i=0; i<3; ++i )
        out<<t[i]<<" ";
    out<<t[3];
    return out;
}


/// @brief enlight a color by a given factor.
RGBAColor RGBAColor::lighten(const RGBAColor& in, const SReal factor)
{
    RGBAColor c = in + ( (RGBAColor::white() - RGBAColor::clamp(in, 0.0f, 1.0f)) * rclamp(float(factor), 0.0f, 1.0f));

    c.a() = 1.0;
    return c ;
}

RGBAColor RGBAColor::operator*(float f) const
{
    return RGBAColor(r() * f, g() * f, b() * f, a() * f);
}


} // namespace sofa::type

