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
#ifndef SOFA_HELPER_STRING_UTILS_H
#define SOFA_HELPER_STRING_UTILS_H

#include <sofa/helper/config.h>

#include <string>
#include <vector>
#include <sstream>

namespace sofa
{

namespace helper
{

///@brief Split one string by a given delimiter and returns that into a std::vector
std::vector<std::string> SOFA_HELPER_API split(const std::string& s, char delimiter);

template<class InputIt, class S>
std::string join(InputIt first, InputIt last, const S& delim)
{
    if(first == last)
        return "";
    std::stringstream ss;
    ss << *first++;
    while(first != last)
    {
        ss << delim << *first++;
    }
    return ss.str();
}

template<class InputIt, class UnaryFunction, class S>
std::string join(InputIt first, InputIt last, UnaryFunction f, const S& delim)
{
    if(first == last)
        return "";
    std::stringstream ss;
    ss << f(*first++);
    while(first != last)
    {
        ss << delim << f(*first++);
    }
    return ss.str();
}

///@brief Join a container into a single string, separated by the provided delimiter.
template<class S, class Container>
std::string join(const Container& elems, const S& delim)
{
    return join(elems.begin(), elems.end(), delim);
}

///@brief returns a copy of the string given in argument.
SOFA_HELPER_API char* getAStringCopy(const char *c);

///@brief replace all occurence of "search" by the "replace" string.
SOFA_HELPER_API void replaceAll(std::string& str,
                                const std::string& search,
                                const std::string& replace);

///@brief returns true if the prefix if located at the beginning of the "full" string.
SOFA_HELPER_API bool starts_with(const std::string& prefix, const std::string& full);

///@brief returns true if the suffix if located at the end of the "full" string.
SOFA_HELPER_API bool ends_with(const std::string& suffix, const std::string& full);

///@brief converts a char* string into a c++ string. The special case with nullptr is coerced to an empty string.
SOFA_HELPER_API std::string safeCharToString(const char* c);

} // namespace helper

} // namespace sofa

#endif //SOFA_HELPER_STRING_UTILS_H
