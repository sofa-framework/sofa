/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "helper.h"

#include <sofa/helper/Logger.h>

#include <clocale>

namespace sofa
{

namespace helper
{

void init()
{
    static bool first = true;
    if (first)
    {
        // Set LC_CTYPE according to the environnement variable, rather than
        // defaulting to "C".  This allows us not to limit ourselves to the
        // 7-bit ASCII character set.  (E.g. see string conversions in
        // helper::Utils).
        char *locale = setlocale(LC_CTYPE, "");
        if (locale == NULL)
            Logger::getMainLogger().log(Logger::Error, "Failed to set LC_CTYPE according to the corresponding environnement variable");

        first = false;
    }
}

} // namespace helper

} // namespace sofa
