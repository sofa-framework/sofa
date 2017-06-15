/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
/*
 * CylinderMesh.cpp
 *
 *  Created on: 12 sep. 2011
 *      Author: Yiyi
 */
#define CGALPLUGIN_CUBOIDMESH_CPP

#include <CGALPlugin/config.h>
#include "CuboidMesh.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

SOFA_DECL_CLASS(CuboidMesh)

using namespace sofa::defaulttype;
using namespace cgal;

int CuboidMeshClass = sofa::core::RegisterObject("Generate a regular tetrahedron mesh of a cuboid")
#ifndef SOFA_FLOAT
        .add< CuboidMesh<Vec3dTypes> >()
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< CuboidMesh<Vec3fTypes> >()
#endif //SOFA_DOUBLE
        ;

#ifndef SOFA_FLOAT
template class SOFA_CGALPLUGIN_API CuboidMesh<Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_CGALPLUGIN_API CuboidMesh<Vec3fTypes>;
#endif //SOFA_DOUBLE
