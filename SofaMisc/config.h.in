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
#ifndef SOFAMISC_CONFIG_H
#define SOFAMISC_CONFIG_H

#include <SofaGeneral/config.h>

#ifdef SOFA_BUILD_MISC_TOPOLOGY
#  define SOFA_TARGET SofaMiscTopology
#  define SOFA_MISC_TOPOLOGY_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_TOPOLOGY_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC_MAPPING
#  define SOFA_TARGET SofaMiscMapping
#  define SOFA_MISC_MAPPING_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_MAPPING_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC_FORCEFIELD
#  define SOFA_TARGET SofaMiscForceField
#  define SOFA_MISC_FORCEFIELD_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_FORCEFIELD_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC_FEM
#  define SOFA_TARGET SofaMiscFem
#  define SOFA_MISC_FEM_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_FEM_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC_ENGINE
#  define SOFA_TARGET SofaMiscEngine
#  define SOFA_MISC_ENGINE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_ENGINE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC_SOLVER
#  define SOFA_TARGET SofaMiscSolver
#  define SOFA_MISC_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MISC
#  define SOFA_TARGET SofaMisc
#  define SOFA_MISC_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MISC_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_COMPONENT_MISC
#  define SOFA_TARGET SofaComponentMisc
#  define SOFA_COMPONENT_MISC_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_COMPONENT_MISC_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif
