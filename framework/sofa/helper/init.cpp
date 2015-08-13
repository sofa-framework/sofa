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
#include "init.h"

#include <sofa/helper/Logger.h>

#include <iostream>

namespace sofa
{

namespace helper
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

SOFA_HELPER_API void init()
{
    if (!s_initialized)
    {
        s_initialized = true;
    }
}

SOFA_HELPER_API bool isInitialized()
{
    return s_initialized;
}

SOFA_HELPER_API void cleanup()
{
    if (!s_cleanedUp)
    {
        s_cleanedUp = true;
    }
}

SOFA_HELPER_API bool isCleanedUp()
{
    return s_cleanedUp;
}

SOFA_HELPER_API void printUninitializedLibraryWarning(const std::string& library,
                                                      const std::string& initFunction)
{
    std::cerr << "Warning: the " << library << " library has not been initialized ("
              << initFunction << " has never been called, see sofa/helper/init.h)" << std::endl;
}

SOFA_HELPER_API void printLibraryNotCleanedUpWarning(const std::string& library,
                                                     const std::string& cleanupFunction)
{
    std::cerr << "Warning: the " << library << " library has not been cleaned up ("
              << cleanupFunction << " has never been called, see sofa/helper/init.h)" << std::endl;
}

// Detect missing cleanup() call.
struct CleanupCheck
{
    ~CleanupCheck()
    {
        if (helper::isInitialized() && !helper::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaHelper", "sofa::helper::cleanup()");
    }
} check;

} // namespace helper

} // namespace sofa
