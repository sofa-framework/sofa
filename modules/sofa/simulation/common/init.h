/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_SIMULATION_COMMON_INIT_H
#define SOFA_SIMULATION_COMMON_INIT_H

#include <sofa/simulation/common/common.h>

namespace sofa
{

namespace simulation
{

namespace common
{

/// @brief Initialize the SofaSimulationCommon library, as well as its
/// dependencies: SofaCore, SofaDefaultType, SofaHelper.
SOFA_SIMULATION_COMMON_API void init();

/// @brief Return true if and only if the SofaSimulationCommon library has been
/// initialized.
SOFA_SIMULATION_COMMON_API bool isInitialized();

/// @brief Clean up the resources used by the SofaSimulationCommon library, as
/// well as its dependencies: SofaCore, SofaDefaultType, SofaHelper.
SOFA_SIMULATION_COMMON_API void cleanup();

/// @brief Return true if and only if the SofaSimulationCommon library has been
/// cleaned up.
SOFA_SIMULATION_COMMON_API bool isCleanedUp();

} // namespace common

} // namespace simulation

} // namespace sofa

#endif
