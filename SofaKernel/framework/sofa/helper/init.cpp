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
#include "init.h"

#include <sofa/helper/system/console.h>
#include <sofa/helper/logging/Messaging.h>

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
        Console::init();
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
    msg_warning(library) << "the library has not been initialized ("
              << initFunction << " has never been called, see sofa/helper/init.h)";
}

SOFA_HELPER_API void printLibraryNotCleanedUpWarning(const std::string& library,
                                                     const std::string& cleanupFunction)
{
    msg_warning(library) << "the library has not been cleaned up ("
              << cleanupFunction << " has never been called, see sofa/helper/init.h)";
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck()
    {
        // to make sure the static variable is created before this
        // and so will be deleted after (at least in c++11)
        // such as an eventual message is possible during this' destructor
        logging::MessageDispatcher::getHandlers();
    }
    ~CleanupCheck()
    {
        if (helper::isInitialized() && !helper::isCleanedUp())
            helper::printLibraryNotCleanedUpWarning("SofaHelper", "sofa::helper::cleanup()");
    }
} check;

} // namespace helper

} // namespace sofa
