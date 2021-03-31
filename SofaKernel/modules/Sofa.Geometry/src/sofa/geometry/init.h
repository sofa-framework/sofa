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
#pragma once

#include <sofa/geometry/config.h>

#include <string>

namespace sofa::geometry
{

/// @brief Initialize the Sofa.Geometry library.
SOFA_GEOMETRY_API void init();

/// @brief Return true if and only if the Sofa.Geometry library has been
/// initialized.
SOFA_GEOMETRY_API bool isInitialized();

/// @brief Clean up the resources used by the Sofa.Geometry library.
SOFA_GEOMETRY_API void cleanup();

/// @brief Return true if and only if the Sofa.Geometry library has been cleaned
/// up.
SOFA_GEOMETRY_API bool isCleanedUp();

/// @brief Print a warning about a library not being initialized (meant for
/// internal use).
SOFA_GEOMETRY_API void printUninitializedLibraryWarning(const std::string& library,
                                                      const std::string& initFunction);

/// @brief Print a warning about a library not being cleaned up (meant for
/// internal use).
SOFA_GEOMETRY_API void printLibraryNotCleanedUpWarning(const std::string& library,
                                                     const std::string& cleanupFunction);

} // namespace sofa::geometry

