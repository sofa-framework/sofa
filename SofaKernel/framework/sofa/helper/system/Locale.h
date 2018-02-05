/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef HELPER_SYSTEM_LOCALE_H
#define HELPER_SYSTEM_LOCALE_H

#include <sofa/helper/helper.h>

#include <clocale>
#include <string>

namespace sofa {

namespace helper {

namespace system {


class SOFA_HELPER_API Locale
{
public:
    static std::string getCategoryName(int category);
};


/// RAII class to modify the locale temporarily.
class SOFA_HELPER_API TemporaryLocale
{
private:
    int m_category;
    std::string m_oldValue;
public:
    TemporaryLocale(int category, std::string locale);
    ~TemporaryLocale();
};


} // namespace system

} // namespace helper

} // sofa

#endif
