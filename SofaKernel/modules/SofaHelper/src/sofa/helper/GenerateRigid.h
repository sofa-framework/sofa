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

#include <sofa/helper/config.h>

#include <sofa/helper/io/Mesh.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Quat.h>

namespace sofa::helper
{

/// base function to compute center of mass, mass and inertia tensor from a mesh
template<typename Rigid3MassType>
void SOFA_HELPER_API generateRigid(Rigid3MassType& mass, type::Vector3& center, const helper::io::Mesh* mesh );

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh, a density, a scale and a rotation
template<typename Rigid3MassType>
SOFA_HELPER_API void generateRigid(Rigid3MassType& mass, type::Vector3& center, io::Mesh *mesh
                                  , SReal density
                                  , const type::Vector3& scale = type::Vector3(1,1,1)
                                  , const type::Vector3& rotation /*Euler angles*/ = type::Vector3(0,0,0)
                                  );

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh file, a density, a scale and a rotation
template<typename Rigid3MassType>
bool SOFA_HELPER_API generateRigid(Rigid3MassType& mass, type::Vector3& center, const std::string& meshFilename
                                  , SReal density
                                  , const type::Vector3& scale = type::Vector3(1,1,1)
                                  , const type::Vector3& rotation /*Euler angles*/ = type::Vector3(0,0,0)
                                  );


/// storing rigid infos needed for RigidMass
struct GenerateRigidInfo
{
    type::Matrix3 inertia;
    type::Quaternion inertia_rotation;
    type::Vector3 inertia_diagonal;
    type::Vector3 com;
    SReal mass;
};

/// user friendly function to compute rigid info from a mesh, a density, a scale
template<typename Rigid3MassType>
void SOFA_HELPER_API generateRigid( GenerateRigidInfo& res
                                  , io::Mesh *mesh
                                  , std::string const& meshName
                                  , SReal density
                                  , const type::Vector3& scale = type::Vector3(1,1,1)
                                  , const type::Vector3& rotation /*Euler angles*/ = type::Vector3(0,0,0)
                                  );

/// user friendly function to compute rigid info from a mesh file, a density, a scale
template<typename Rigid3MassType>
bool SOFA_HELPER_API generateRigid( GenerateRigidInfo& res
                                  , const std::string& meshFilename
                                  , SReal density
                                  , const type::Vector3& scale = type::Vector3(1,1,1)
                                  , const type::Vector3& rotation /*Euler angles*/ = type::Vector3(0,0,0)
                                  );

} // namespace sofa::helper
