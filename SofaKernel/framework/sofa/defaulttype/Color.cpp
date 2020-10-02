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
#define SOFA_DEFAULTTYPE_COLOR_CPP
#include <sofa/defaulttype/Color.h>

namespace sofa
{
namespace defaulttype
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


SOFA_DEFAULTTYPE_API std::istream& operator>>(std::istream& i, RGBAColor& t)
{
    std::string s;
    std::getline(i, s);
    if(!RGBAColor::read(s, t)){
        i.setstate(std::ios::failbit) ;
    }

    return i;
}

} // namespace defaulttype
} // namespace sofa

