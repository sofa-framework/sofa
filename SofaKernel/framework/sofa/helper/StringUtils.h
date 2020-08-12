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
#ifndef SOFA_HELPER_STRING_UTILS_H
#define SOFA_HELPER_STRING_UTILS_H

#include <string>
#include <cstring>
#include <vector>
#include <sstream>

namespace sofa
{

namespace helper
{

/// Taken from https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
static inline std::vector<std::string> split(const std::string& s, char delimiter)
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

/// Taken from https://github.com/ekg/split/blob/master/join.h (I don't know what is the licence
/// but thank for the author.
template<class S, class T>
std::string join(std::vector<T>& elems, S& delim) {
    std::stringstream ss;
    if(elems.empty())
        return "";
    typename std::vector<T>::iterator e = elems.begin();
    ss << *e++;
    for (; e != elems.end(); ++e) {
        ss << delim << *e;
    }
    return ss.str();
}

static inline char* getAStringCopy(const char *c)
{
    char* tmp = new char[strlen(c)+1] ;
    strcpy(tmp,c);
    return tmp ;
}

static inline void replaceAll(std::string& str, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while((pos = str.find(search, pos)) != std::string::npos)
    {
        str.replace(pos, search.length(), replace);
        pos += replace.length();
    }
}

} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_STRING_UTILS_H
