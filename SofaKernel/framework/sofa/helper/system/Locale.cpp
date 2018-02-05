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
#include "Locale.h"
#include <sofa/helper/logging/Messaging.h>
#include <clocale>

namespace sofa {

namespace helper {

namespace system {


TemporaryLocale::TemporaryLocale(int category, std::string locale):
    m_category(category), m_oldValue(std::string(setlocale(category, NULL)))
{
    char *value = setlocale(category, locale.c_str());
    if (value == NULL)
        msg_error("TemporaryLocale") << "Failed to set " << Locale::getCategoryName(category) << " to " << locale;
}

TemporaryLocale::~TemporaryLocale()
{
    setlocale(m_category, m_oldValue.c_str());
}

std::string Locale::getCategoryName(int category)
{
    switch(category)
    {
    case LC_ALL:
        return "LC_ALL";
    case LC_COLLATE:
        return "LC_COLLATE";
    case LC_CTYPE:
        return "LC_CTYPE";
#if WIN32
#if (_MSC_VER < 1800)	// visual studio >= 2013 does not recognize LC_MESSAGES
    case LC_MESSAGES:
        return "LC_MESSAGES";
#endif
#else
	case LC_MESSAGES:
		return "LC_MESSAGES";
#endif
    case LC_MONETARY:
        return "LC_MONETARY";
    case LC_NUMERIC:
        return "LC_NUMERIC";
    case LC_TIME:
        return "LC_TIME";
    default:
        return "UNKNOWN";
    }
}


} // namespace system

} // namespace helper

} // sofa
