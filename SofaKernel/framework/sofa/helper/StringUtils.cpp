/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <cstring>
#include "StringUtils.h"

namespace sofa
{

namespace helper
{

/// Taken from https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

char* getAStringCopy(const char *c)
{
    char* tmp = new char[strlen(c)+1] ;
    strcpy(tmp,c);
    return tmp ;
}

void replaceAll(std::string& str, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = str.find(search, pos)) != std::string::npos)
    {
        str.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

bool ends_with(const std::string& suffix, const std::string& full)
{
    const std::size_t lf = full.length();
    const std::size_t ls = suffix.length();

    if(lf < ls) return false;

    return (0 == full.compare(lf - ls, ls, suffix));
}

bool starts_with(const std::string& prefix, const std::string& full)
{
    const std::size_t lf = full.length();
    const std::size_t lp = prefix.length();

    if(lf < lp) return false;

    return (0 == full.compare(0, lp, prefix));
}

} // namespace helper

} // namespace sofa

