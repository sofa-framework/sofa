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
#include <stack>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/NameDecoder.h>
#include <cctype>

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
    std::string typeName;
    const std::string realname = NameDecoder::decodeFullName(t);
    const size_t len = realname.length();
    typeName.resize(len+1);
    size_t start = 0;
    size_t dest = 0;
    size_t chevron {}; //count the number of opening chevrons: indicates the level of nested templates

    for (size_t i=0; i<len; i++)
    {
        const char c = realname[i];

        if (c == '<')
            ++chevron;
        else if (c == '>')
            --chevron;

        if (c == ':')
        {
            if (chevron == 0)
                dest = 0; //restart when a semi-column is met outside from a template parameter
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
        else if (c == ' ' && ((i >= 1 && !std::isalnum(realname[i-1]))  || (i < len - 1 && !std::isalnum(realname[i+1])) ) )
        {
            //skip space if character before or after is not alphanumeric
            start = i+1;
        }
        else if (c != '_' && !std::isalnum(c))
        {
            // write result
            while (start <= i)
            {
                typeName[dest++] = realname[start++];
            }
        }
    }
    while (start < len)
    {
        typeName[dest++] = realname[start++];
    }
    typeName.resize(dest);
    return typeName;
}

std::string NameDecoder::decodeClassName(const std::type_info& t)
{
    std::string className;
    const std::string realname = NameDecoder::decodeFullName(t);
    const size_t len = realname.length();
    className.resize(len+1);
    size_t start = 0;
    size_t dest = 0;
    size_t i;
    size_t chevron {}; //count the number of opening chevrons: indicates the level of nested templates

    for (i = 0; i < len; ++i)
    {
        const char c = realname[i];

        if (c == '<')
        {
            if (++chevron == 1)
            {
                //copy string when it's the first opening chevron
                while (start < i)
                {
                    className[dest++] = realname[start++];
                }
            }
        }
        else if (c == '>')
        {
            start = i+1;
            --chevron;
        }

        if (chevron > 0)
        {
            start = i+1;
            continue;
        }

        if (c == ':')
        {
            if (chevron == 0)
                dest = 0; //restart when a semi-column is met outside from a template parameter
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
        else if (c == ' ' && ((i >= 1 && !std::isalnum(realname[i-1]))  || (i < len - 1 && !std::isalnum(realname[i+1])) ) )
        {
            //skip space if character before or after is not alphanumeric
            start = i+1;
        }
        else if (c != '_' && !std::isalnum(c))
        {
            while (start < i)
            {
                className[dest++] = realname[start++];
            }
        }
    }

    while (start < i)
    {
        className[dest++] = realname[start++];
    }
    className.resize(dest);
    return className;
}

std::string NameDecoder::decodeNamespaceName(const std::type_info& t)
{
    const std::string realname = NameDecoder::decodeFullName(t);
    const size_t len = realname.length();
    size_t start = 0;
    size_t last = len-1;
    for (size_t i = 0; i<len; i++)
    {
        const char c = realname[i];
        if (c == ' ' && i >= 5 && realname[i-5] == 'c' && realname[i-4] == 'l' && realname[i-3] == 'a' && realname[i-2] == 's' && realname[i-1] == 's')
        {
            start = i+1;
        }
        else if (c == ' ' && i >= 6 && realname[i-6] == 's' && realname[i-5] == 't' && realname[i-4] == 'r' && realname[i-3] == 'u' && realname[i-2] == 'c' && realname[i-1] == 't')
        {
            start = i+1;
        }
        else if (c == ' ' && ((i >= 1 && !std::isalnum(realname[i-1]))  || (i < len - 1 && !std::isalnum(realname[i+1])) ) )
        {
            //skip space if character before or after is not alphanumeric
            start = i+1;
        }
        else if (c == ':' && (i<1 || realname[i-1]!=':'))
        {
            last = i-1;
        }
        else if (c != ':' && c != '_' && !std::isalnum(c))
        {
            // write result
            break;
        }
    }
    if (last == len-1)
        return {};
    std::string namespaceName = realname.substr(start, last - start + 1);
    return namespaceName;
}

std::string NameDecoder::decodeTemplateName(const std::type_info& t)
{
    const std::string realname = NameDecoder::decodeFullName(t);
    const size_t len = realname.length();

    size_t i = realname.find_first_of('<');

    if (i == std::string::npos)
    {
        return {};
    }

    std::string templateName;
    templateName.resize(len - i + 1);
    size_t dest = 0;

    size_t start = i + 1;
    ++i;

    size_t chevron = 1; // count the number of opening chevrons: indicates the level of nested templates

    for (; i<len; i++)
    {
        const char c = realname[i];

        if (c == '<')
        {
            if (++chevron == 1)
            {
                dest = 0; // restart when starting a template parameter
                start = i+1;
            }
        }
        else if (c == '>')
            --chevron;

        if (c == ':')
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
        else if (c == ' ' && ((i >= 1 && !std::isalnum(realname[i-1])) || (i < len - 1 && !std::isalnum(realname[i+1])) ) )
        {
            //skip space if character before or after is not alphanumeric
            start = i+1;
        }
        else if (c == ',') //template separator
        {
            while (start <= i)
            {
                templateName[dest++] = realname[start++];
            }
        }
        else if (c != '_' && !std::isalnum(c))
        {
            // write result
            while (start < i + (chevron > 0))
            {
                templateName[dest++] = realname[start++];
            }
        }
    }

    templateName.resize(dest);
    return templateName;
}


}
