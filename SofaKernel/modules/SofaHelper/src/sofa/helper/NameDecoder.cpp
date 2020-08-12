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
#include <sofa/helper/logging/Messaging.h>
#include "NameDecoder.h"

#ifdef __GNUC__
#include <cxxabi.h>
#endif // __GNUC__

namespace sofa::helper
{

std::string NameDecoder::shortName( const std::string& src )
{
    if( src.empty() )
        return "";
    std::string dst=src;
    *dst.begin() = char(::tolower(*dst.begin()));
    return  dst;
}


std::string NameDecoder::decodeFullName(const std::type_info& t)
{
    std::string name;
#ifdef __GNUC__
    int status;
    /* size_t length; */ // although it should, length would not be filled in by the following call
    char* allocname = abi::__cxa_demangle(t.name(), nullptr, /*&length*/nullptr, &status);
    if(allocname == nullptr)
    {
        msg_error("BaseClass") << "decodeFullName: Unable to demangle symbol: " << t.name();
    }
    else
    {
        size_t length = 0;
        while(allocname[length] != '\0')
        {
            length++;
        }
        name.resize(length);
        for(size_t i=0; i < length; i++)
            name[i] = allocname[i];
        free(allocname);
    }
#else
    name = t.name();
#endif
    return name;
}

std::string NameDecoder::decodeTypeName(const std::type_info& t)
{
    std::string name;
    std::string realname = NameDecoder::decodeFullName(t);
    size_t len = realname.length();
    name.resize(len+1);
    size_t start = 0;
    size_t dest = 0;
    //char cprev = '\0';
    for (size_t i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9'))
        {
            // write result
            while (start < i)
            {
                name[dest++] = realname[start++];
            }
        }
        //cprev = c;
    }
    while (start < len)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());  
    return name;
}

std::string NameDecoder::decodeClassName(const std::type_info& t)
{
    std::string name;
    std::string realname = NameDecoder::decodeFullName(t);
    size_t len = realname.length();
    name.resize(len+1);
    size_t start = 0;
    size_t dest = 0;
    size_t i;
    //char cprev = '\0';

    for (i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == '<') break;
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9'))
        {
            // write result
            while (start < i)
            {
                name[dest++] = realname[start++];
            }
        }
        //cprev = c;
    }

    while (start < i)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());  
    return name;
}

std::string NameDecoder::decodeNamespaceName(const std::type_info& t)
{
    std::string name;
    std::string realname = NameDecoder::decodeFullName(t);
    size_t len = realname.length();
    size_t start = 0;
    size_t last = len-1;
    size_t i;
    for (i=0; i<len; i++)
    {
        char c = realname[i];
        if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c == ':' && (i<1 || realname[i-1]!=':'))
        {
            last = i-1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9'))
        {
            // write result
            break;
        }
    }
    name = realname.substr(start, last-start+1);
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());  
    return name;
}

std::string NameDecoder::decodeTemplateName(const std::type_info& t)
{
    std::string name;
    std::string realname = NameDecoder::decodeFullName(t);
    size_t len = realname.length();
    name.resize(len+1);
    size_t start = 0;
    size_t dest = 0;
    size_t i = 0;
    //char cprev = '\0';
    while (i < len && realname[i]!='<')
        ++i;
    start = i+1; ++i;
    for (; i<len; i++)
    {
        char c = realname[i];
        //if (c == '<') break;
        if (c == ':') // && cprev == ':')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c != ':' && c != '_' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9'))
        {
            // write result
            while (start <= i)
            {
                name[dest++] = realname[start++];
            }
        }
        //cprev = c;
    }
    while (start < i)
    {
        name[dest++] = realname[start++];
    }
    name.resize(dest);
    name.erase(std::remove(name.begin(), name.end(), ' '), name.end());  
    return name;
}


}
