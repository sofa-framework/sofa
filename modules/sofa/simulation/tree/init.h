/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_TREE_INIT_H
#define SOFA_SIMULATION_TREE_INIT_H

#include <sofa/simulation/tree/tree.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

/// @brief Initialize the SofaSimulationTree library, as well as its
/// dependencies: SofaSimulationCommon, SofaCore, SofaDefaultType, SofaHelper.
void SOFA_SIMULATION_TREE_API init();

/// @brief Return true if and only if the SofaSimulationTree library has been
/// initialized.
bool SOFA_SIMULATION_TREE_API isInitialized();

/// @brief Clean up the resources used by the SofaSimulationTree library, as
/// well as its dependencies: SofaSimulationCommon, SofaCore, SofaDefaultType,
/// SofaHelper.
void SOFA_SIMULATION_TREE_API cleanup();

/// @brief Return true if and only if the SofaSimulationTree library has been
/// cleaned up.
bool SOFA_SIMULATION_TREE_API isCleanedUp();

/// @brief Print a warning if the SofaSimulationTree library is not
/// initialized.
void SOFA_SIMULATION_TREE_API checkIfInitialized();

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
