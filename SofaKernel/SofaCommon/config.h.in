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
#ifndef SOFACOMMON_CONFIG_H
#define SOFACOMMON_CONFIG_H

#include <SofaBase/config.h>

#ifdef SOFA_BUILD_LOADER
#  define  SOFA_TARGET    SofaLoader
#  define SOFA_LOADER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_LOADER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_RIGID
#  define SOFA_TARGET    SofaRigid
#  define SOFA_RIGID_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_RIGID_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_DEFORMABLE
#  define SOFA_TARGET    SofaDeformable
#  define SOFA_DEFORMABLE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_DEFORMABLE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_SIMPLE_FEM
#  define SOFA_TARGET    SofaSimpleFem
#  define SOFA_SIMPLE_FEM_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_SIMPLE_FEM_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_OBJECT_INTERACTION
#  define SOFA_TARGET    SofaObjectInteraction
#  define SOFA_OBJECT_INTERACTION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_OBJECT_INTERACTION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_MESH_COLLISION
#  define SOFA_TARGET    SofaMeshCollision
#  define SOFA_MESH_COLLISION_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_MESH_COLLISION_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_EXPLICIT_ODE_SOLVER
#  define SOFA_TARGET    SofaExplicitOdeSolver
#  define SOFA_EXPLICIT_ODE_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_EXPLICIT_ODE_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_IMPLICIT_ODE_SOLVER
#  define SOFA_TARGET    SofaImplicitOdeSolver
#  define SOFA_IMPLICIT_ODE_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_IMPLICIT_ODE_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_ENGINE
#  define SOFA_TARGET SofaEngine
#  define SOFA_ENGINE_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_ENGINE_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_EIGEN2_SOLVER
#  define SOFA_TARGET SofaEigen2Solver
#  define SOFA_EIGEN2_SOLVER_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_EIGEN2_SOLVER_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#ifdef SOFA_BUILD_COMPONENT_COMMON
#  define SOFA_TARGET    SofaComponentCommon
#  define SOFA_COMPONENT_COMMON_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_COMPONENT_COMMON_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif


#endif
