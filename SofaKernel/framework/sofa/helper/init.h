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
#ifndef SOFA_HELPER_INIT_H
#define SOFA_HELPER_INIT_H

#include <sofa/helper/helper.h>

#include <string>

namespace sofa
{

namespace helper
{

// Initializing and cleaning up Sofa
//
// The Sofa framework is split into multiple libraries.  In order to initialize
// all the Sofa libraries it uses, an application only needs to call the init()
// function of the "higher level" libraries it links against, which will call
// the init() function of their dependencies, like so:
//
//       "Higher level"
//     ------------------>
//
//     sofa::helper::init()                             // SofaHelper
//     └──sofa::defaulttype::init()                     // SofaDefaultType
//        └──sofa::core::init()                         // SofaCore
//           └──sofa::simulation::common::init()        // SofaSimulationCommon
//              ├──sofa::simulation::tree::init()       // SofaSimulationTree
//              └──sofa::simulation::graph::init()      // SofaSimulationGraph
//
// For example:
//
// - If an application links against SofaSimulationTree, it only needs to call
//   sofa::simulation::tree::init().
//
// - If it links against SofaCore, it only needs to call sofa::core::init().
//
// - If it links against both SofaSimulationTree and SofaSimulationGraph, it
//   needs to call both sofa::simulation::tree::init() and
//   sofa::simulation::graph::init().
//
//
// Symmetrically, before exiting, an application needs to call the cleanup()
// function of the libraries it init()'ed.

/// @brief Initialize the SofaHelper library.
SOFA_HELPER_API void init();

/// @brief Return true if and only if the SofaHelper library has been
/// initialized.
SOFA_HELPER_API bool isInitialized();

/// @brief Clean up the resources used by the SofaHelper library.
SOFA_HELPER_API void cleanup();

/// @brief Return true if and only if the SofaHelper library has been cleaned
/// up.
SOFA_HELPER_API bool isCleanedUp();

/// @brief Print a warning about a library not being initialized (meant for
/// internal use).
SOFA_HELPER_API void printUninitializedLibraryWarning(const std::string& library,
                                                      const std::string& initFunction);

/// @brief Print a warning about a library not being cleaned up (meant for
/// internal use).
SOFA_HELPER_API void printLibraryNotCleanedUpWarning(const std::string& library,
                                                     const std::string& cleanupFunction);

} // namespace helper

} // namespace sofa

#endif
