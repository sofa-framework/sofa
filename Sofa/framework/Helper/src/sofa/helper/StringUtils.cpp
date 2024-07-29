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
#include <cstring>
#include <sofa/helper/StringUtils.h>
#include <algorithm>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/Locale.h>


namespace sofa::helper
{

std::wstring widenString(const std::string& s)
{
    // Set LC_CTYPE according to the environnement variable, for mbsrtowcs().
    system::TemporaryLocale locale(LC_CTYPE, "");

    const char * src = s.c_str();
    // Call mbsrtowcs() once to find out the length of the converted string.
    size_t length = mbsrtowcs(nullptr, &src, 0, nullptr);
    if (length == size_t(-1)) {
        const int error = errno;
        msg_warning("Utils::widenString()") << strerror(error);
        return L"";
    }

    // Call mbsrtowcs() again with a correctly sized buffer to actually do the conversion.
    wchar_t * buffer = new wchar_t[length + 1];
    length = mbsrtowcs(buffer, &src, length + 1, nullptr);
    if (length == size_t(-1)) {
        const int error = errno;
        msg_warning("Utils::widenString()") << strerror(error);
        delete[] buffer;
        return L"";
    }

    if (src != nullptr) {
        msg_warning("Utils::widenString()") << "Conversion failed (\"" << s << "\")";
        delete[] buffer;
        return L"";
    }

    std::wstring result(buffer);
    delete[] buffer;
    return result;
}

std::string narrowString(const std::wstring& ws)
{
    // Set LC_CTYPE according to the environnement variable, for wcstombs().
    system::TemporaryLocale locale(LC_CTYPE, "");

    const wchar_t * src = ws.c_str();
    // Call wcstombs() once to find out the length of the converted string.
    size_t length = wcstombs(nullptr, src, 0);
    if (length == size_t(-1)) {
        msg_warning("Utils::narrowString()") << "Conversion failed";
        return "";
    }

    // Call wcstombs() again with a correctly sized buffer to actually do the conversion.
    char * buffer = new char[length + 1];
    length = wcstombs(buffer, src, length + 1);
    if (length == size_t(-1)) {
        msg_warning("Utils::narrowString()") << "Conversion failed";
        delete[] buffer;
        return "";
    }

    std::string result(buffer);
    delete[] buffer;
    return result;
}

std::string downcaseString(const std::string& s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::string upcaseString(const std::string& s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

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

std::string safeCharToString(const char* c)
{
    if(c==nullptr)
        return std::string("");
    return std::string(c);
}

std::string_view removeTrailingCharacter(std::string_view sv, char character)
{
    auto end = sv.end();
    while (end != sv.begin() && *(end - 1) == character)
    {
        --end;
    }
    return sv.substr(0, end - sv.begin());
}

std::string_view removeTrailingCharacters(std::string_view sv, const std::initializer_list<char> characters)
{
    auto end = sv.end();
    while (end != sv.begin() && std::find(characters.begin(), characters.end(), *(end - 1)) != characters.end())
    {
        --end;
    }
    return sv.substr(0, end - sv.begin());
}

} // namespace sofa::helper



