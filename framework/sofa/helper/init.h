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
#ifndef SOFA_HELPER_INIT_H
#define SOFA_HELPER_INIT_H

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

/// @brief Initialize the SofaHelper library.
void SOFA_HELPER_API init();

/// @brief Return true if and only if the SofaHelper library has been
/// initialized.
bool SOFA_HELPER_API isInitialized();

/// @brief Clean up the resources used by the SofaHelper library.
void SOFA_HELPER_API cleanup();

/// @brief Return true if and only if the SofaHelper library has been cleaned
/// up.
bool SOFA_HELPER_API isCleanedUp();

/// @brief Print a warning if the SofaHelper library is not initialized.
void SOFA_HELPER_API checkIfInitialized();

} // namespace helper

} // namespace sofa

#endif
