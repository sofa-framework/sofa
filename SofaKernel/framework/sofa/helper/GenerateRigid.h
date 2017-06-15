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
#ifndef GENERATERIGID_H
#define GENERATERIGID_H

#include "sofa/config.h"

#include <sofa/helper/io/Mesh.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace helper
{

/// base function to compute center of mass, mass and inertia tensor from a mesh
void SOFA_HELPER_API generateRigid( defaulttype::Rigid3Mass& mass, defaulttype::Vector3& center, const helper::io::Mesh* mesh );

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh, a density, a scale and a rotation
void generateRigid(defaulttype::Rigid3Mass& mass, defaulttype::Vector3& center, io::Mesh *mesh
                                  , SReal density
                                  , const defaulttype::Vector3& scale = defaulttype::Vector3(1,1,1)
                                  , const defaulttype::Vector3& rotation /*Euler angles*/ = defaulttype::Vector3(0,0,0)
                                  );

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh file, a density, a scale and a rotation
bool SOFA_HELPER_API generateRigid( defaulttype::Rigid3Mass& mass, defaulttype::Vector3& center, const std::string& meshFilename
                                  , SReal density
                                  , const defaulttype::Vector3& scale = defaulttype::Vector3(1,1,1)
                                  , const defaulttype::Vector3& rotation /*Euler angles*/ = defaulttype::Vector3(0,0,0)
                                  );


/// storing rigid infos needed for RigidMass
struct GenerateRigidInfo
{
    defaulttype::Matrix3 inertia;
    defaulttype::Quaternion inertia_rotation;
    defaulttype::Vector3 inertia_diagonal;
    defaulttype::Vector3 com;
    SReal mass;
};

/// user friendly function to compute rigid info from a mesh, a density, a scale
void SOFA_HELPER_API generateRigid( GenerateRigidInfo& res
                                  , io::Mesh *mesh
                                  , std::string const& meshName
                                  , SReal density
                                  , const defaulttype::Vector3& scale = defaulttype::Vector3(1,1,1)
                                  , const defaulttype::Vector3& rotation /*Euler angles*/ = defaulttype::Vector3(0,0,0)
                                  );

/// user friendly function to compute rigid info from a mesh file, a density, a scale
bool SOFA_HELPER_API generateRigid( GenerateRigidInfo& res
                                  , const std::string& meshFilename
                                  , SReal density
                                  , const defaulttype::Vector3& scale = defaulttype::Vector3(1,1,1)
                                  , const defaulttype::Vector3& rotation /*Euler angles*/ = defaulttype::Vector3(0,0,0)
                                  );
}

}

#endif

