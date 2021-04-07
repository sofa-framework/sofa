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
#include <sofa/topology/init.h>

#include <iostream>

namespace sofa::topology
{

static bool s_initialized = false;
static bool s_cleanedUp = false;

SOFA_TOPOLOGY_API void init()
{
    if (!s_initialized)
    {
        s_initialized = true;
    }
}

SOFA_TOPOLOGY_API bool isInitialized()
{
    return s_initialized;
}

SOFA_TOPOLOGY_API void cleanup()
{
    if (!s_cleanedUp)
    {
        s_cleanedUp = true;
    }
}

SOFA_TOPOLOGY_API bool isCleanedUp()
{
    return s_cleanedUp;
}

SOFA_TOPOLOGY_API void printUninitializedLibraryWarning(const std::string& library,
                                                      const std::string& initFunction)
{
    std::cerr << "WARNING: " << library << " : the library has not been initialized ("
              << initFunction << " has never been called, see sofa/topology/init.h)"
              << std::endl;
}

SOFA_TOPOLOGY_API void printLibraryNotCleanedUpWarning(const std::string& library,
                                                     const std::string& cleanupFunction)
{
    std::cerr << "WARNING: " << library << " : the library has not been cleaned up ("
              << cleanupFunction << " has never been called, see sofa/topology/init.h)"
              << std::endl;
}

// Detect missing cleanup() call.
static const struct CleanupCheck
{
    CleanupCheck()
    {

    }
    ~CleanupCheck()
    {
        if (topology::isInitialized() && !topology::isCleanedUp())
            topology::printLibraryNotCleanedUpWarning("Sofa.Topology", "sofa::topology::cleanup()");
    }
} check;

} // namespace sofa::topology
