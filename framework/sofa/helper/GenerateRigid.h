/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef GENERATERIGID_H
#define GENERATERIGID_H

#include <sofa/helper/io/Mesh.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace helper
{

/// base function to compute center of mass, mass and inertia tensor from a mesh
void SOFA_HELPER_API GenerateRigid( defaulttype::Rigid3Mass& mass, defaulttype::Vector3& center, const helper::io::Mesh* mesh );

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh file, a density, a scale and a rotation
bool SOFA_HELPER_API GenerateRigid( defaulttype::Rigid3Mass& mass, defaulttype::Vector3& center, const std::string& meshFilename
                  , SReal density
                  , const defaulttype::Vector3& scale = defaulttype::Vector3(1,1,1)
                  , const defaulttype::Vector3& rotation /*Euler angles*/ = defaulttype::Vector3(0,0,0)
                  );

}

}

#endif
