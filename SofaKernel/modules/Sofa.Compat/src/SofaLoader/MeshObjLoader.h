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

#include <SofaLoader/MeshOBJLoader.h>

// on Windows, we will never be sure that this compat file will be used instead of the legit MeshOBJLoader.h one.
// Because the filesystem is usually case-insensitive, and it will make no difference between 
// <SofaLoader/MeshObjLoader.h> and <SofaLoader/MeshOBJLoader.h>
// It can happen with MacOS as well.
SOFA_DEPRECATED_HEADER("v21.12", "v22.12", "SofaLoader/MeshOBJLoader.h")

using MeshObjLoader SOFA_ATTRIBUTE_DEPRECATED("v21.12 (PR#2428)", "v22.12", "MeshObjLoader has been renamed to MeshOBJLoader") = MeshOBJLoader;
