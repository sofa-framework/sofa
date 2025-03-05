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
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/decompose.h>

namespace sofa::helper
{

/// base function to compute center of mass, mass and inertia tensor from a mesh
template<typename Rigid3MassType>
void generateRigid(Rigid3MassType& mass, type::Vec3& center, const helper::io::Mesh* mesh )
{
    using namespace sofa::helper;
    // Geometric Tools, Inc.
    // http://www.geometrictools.com
    // Copyright (c) 1998-2006.	All Rights Reserved
    //
    // The Wild Magic Library (WM3) source code is supplied under the terms of
    // the license agreement
    //     http://www.geometrictools.com/License/WildMagic3License.pdf
    // and may not be copied or disclosed except in accordance with the terms
    // of that agreement.

    // order:	1, x, y, z, x^2, y^2, z^2, xy, yz, zx
    SReal afIntegral[10] = { (SReal)0.0, (SReal)0.0, (SReal)0.0, (SReal)0.0,
            (SReal)0.0, (SReal)0.0, (SReal)0.0, (SReal)0.0, (SReal)0.0, (SReal)0.0
                            };

    const auto& points = mesh->getVertices();
    const auto& facets = mesh->getFacets();
    for (unsigned int i = 0; i < facets.size(); i++)
    {
        const auto& v = facets[i][0];
        for (unsigned int j = 2; j < v.size(); j++)
        {
            // get vertices of current triangle
            const auto kV0 = points[v[0  ]];
            const auto kV1 = points[v[j-1]];
            const auto kV2 = points[v[j  ]];

            // get cross product of edges
            auto kV1mV0 = kV1 - kV0;
            auto kV2mV0 = kV2 - kV0;
            auto kN = cross(kV1mV0,kV2mV0);

            // compute integral terms
            SReal fTmp0, fTmp1, fTmp2;
            SReal fF1x, fF2x, fF3x, fG0x, fG1x, fG2x;
            fTmp0 = kV0[0] + kV1[0];
            fF1x = fTmp0 + kV2[0];
            fTmp1 = kV0[0]*kV0[0];
            fTmp2 = fTmp1 + kV1[0]*fTmp0;
            fF2x = fTmp2 + kV2[0]*fF1x;
            fF3x = kV0[0]*fTmp1 + kV1[0]*fTmp2 + kV2[0]*fF2x;
            fG0x = fF2x + kV0[0]*(fF1x + kV0[0]);
            fG1x = fF2x + kV1[0]*(fF1x + kV1[0]);
            fG2x = fF2x + kV2[0]*(fF1x + kV2[0]);

            SReal fF1y, fF2y, fF3y, fG0y, fG1y, fG2y;
            fTmp0 = kV0[1] + kV1[1];
            fF1y = fTmp0 + kV2[1];
            fTmp1 = kV0[1]*kV0[1];
            fTmp2 = fTmp1 + kV1[1]*fTmp0;
            fF2y = fTmp2 + kV2[1]*fF1y;
            fF3y = kV0[1]*fTmp1 + kV1[1]*fTmp2 + kV2[1]*fF2y;
            fG0y = fF2y + kV0[1]*(fF1y + kV0[1]);
            fG1y = fF2y + kV1[1]*(fF1y + kV1[1]);
            fG2y = fF2y + kV2[1]*(fF1y + kV2[1]);

            SReal fF1z, fF2z, fF3z, fG0z, fG1z, fG2z;
            fTmp0 = kV0[2] + kV1[2];
            fF1z = fTmp0 + kV2[2];
            fTmp1 = kV0[2]*kV0[2];
            fTmp2 = fTmp1 + kV1[2]*fTmp0;
            fF2z = fTmp2 + kV2[2]*fF1z;
            fF3z = kV0[2]*fTmp1 + kV1[2]*fTmp2 + kV2[2]*fF2z;
            fG0z = fF2z + kV0[2]*(fF1z + kV0[2]);
            fG1z = fF2z + kV1[2]*(fF1z + kV1[2]);
            fG2z = fF2z + kV2[2]*(fF1z + kV2[2]);

            // update integrals
            afIntegral[0] += kN[0]*fF1x;
            afIntegral[1] += kN[0]*fF2x;
            afIntegral[2] += kN[1]*fF2y;
            afIntegral[3] += kN[2]*fF2z;
            afIntegral[4] += kN[0]*fF3x;
            afIntegral[5] += kN[1]*fF3y;
            afIntegral[6] += kN[2]*fF3z;
            afIntegral[7] += kN[0]*(kV0[1]*fG0x + kV1[1]*fG1x + kV2[1]*fG2x);
            afIntegral[8] += kN[1]*(kV0[2]*fG0y + kV1[2]*fG1y + kV2[2]*fG2y);
            afIntegral[9] += kN[2]*(kV0[0]*fG0z + kV1[0]*fG1z + kV2[0]*fG2z);
        }
    }

    afIntegral[0] /= 6.0_sreal;
    afIntegral[1] /= 24.0_sreal;
    afIntegral[2] /= 24.0_sreal;
    afIntegral[3] /= 24.0_sreal;
    afIntegral[4] /= 60.0_sreal;
    afIntegral[5] /= 60.0_sreal;
    afIntegral[6] /= 60.0_sreal;
    afIntegral[7] /= 120.0_sreal;
    afIntegral[8] /= 120.0_sreal;
    afIntegral[9] /= 120.0_sreal;

    // mass
    mass.volume = afIntegral[0];
    mass.mass = mass.volume;

    // center of mass
    center = {afIntegral[1]/afIntegral[0],afIntegral[2]/afIntegral[0],afIntegral[3]/afIntegral[0]};

    // inertia relative to world origin
    mass.inertiaMatrix[0][0] = afIntegral[5] + afIntegral[6];
    mass.inertiaMatrix[0][1] = -afIntegral[7];
    mass.inertiaMatrix[0][2] = -afIntegral[9];
    mass.inertiaMatrix[1][0] = mass.inertiaMatrix[0][1];
    mass.inertiaMatrix[1][1] = afIntegral[4] + afIntegral[6];
    mass.inertiaMatrix[1][2] = -afIntegral[8];
    mass.inertiaMatrix[2][0] = mass.inertiaMatrix[0][2];
    mass.inertiaMatrix[2][1] = mass.inertiaMatrix[1][2];
    mass.inertiaMatrix[2][2] = afIntegral[4] + afIntegral[5];

    // inertia relative to center of mass
    mass.inertiaMatrix[0][0] -= mass.mass*(center[1]*center[1] + center[2]*center[2]);
    mass.inertiaMatrix[0][1] += mass.mass*center[0]*center[1];
    mass.inertiaMatrix[0][2] += mass.mass*center[2]*center[0];
    mass.inertiaMatrix[1][0] = mass.inertiaMatrix[0][1];
    mass.inertiaMatrix[1][1] -= mass.mass*(center[2]*center[2] + center[0]*center[0]);
    mass.inertiaMatrix[1][2] += mass.mass*center[1]*center[2];
    mass.inertiaMatrix[2][0] = mass.inertiaMatrix[0][2];
    mass.inertiaMatrix[2][1] = mass.inertiaMatrix[1][2];
    mass.inertiaMatrix[2][2] -= mass.mass*(center[0]*center[0] + center[1]*center[1]);

    mass.inertiaMatrix /= mass.mass;
}

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh, a density, a scale and a rotation
template<typename Rigid3MassType>
void generateRigid(Rigid3MassType& mass, type::Vec3& center, io::Mesh *mesh
                                  , SReal density
                                  , const type::Vec3& scale = type::Vec3(1,1,1)
                                  , const type::Vec3& rotation /*Euler angles*/ = type::Vec3(0,0,0)
                                  )
{
    if( scale != type::Vec3(1, 1, 1) ) {
        for(size_t i = 0, n = mesh->getVertices().size(); i < n; ++i) {
            mesh->getVertices()[i] = mesh->getVertices()[i].linearProduct(scale);
        }
    }

    if( rotation != type::Vec3(0,0,0) ) {
        const type::Quat q = type::Quat<SReal>::createQuaterFromEuler( rotation*M_PI/180.0 );

        for(size_t i = 0, n = mesh->getVertices().size(); i < n; ++i) {
            mesh->getVertices()[i] = q.rotate( mesh->getVertices()[i] );
        }
    }

    generateRigid( mass, center, mesh );

    mass.recalc();

    mass.mass *= density;

}

/// user friendly function to compute center of mass, mass and inertia tensor from a mesh file, a density, a scale and a rotation
template<typename Rigid3MassType>
bool generateRigid(Rigid3MassType& mass, type::Vec3& center, const std::string& meshFilename
                                  , SReal density
                                  , const type::Vec3& scale = type::Vec3(1,1,1)
                                  , const type::Vec3& rotation /*Euler angles*/ = type::Vec3(0,0,0)
                                  )
{
    sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create( meshFilename );
    if (mesh == nullptr)
    {
        msg_error("GenerateRigid") << "unable to loading mesh from file '"<<meshFilename<<"'" ;
        return false;
    }

    generateRigid(mass, center, mesh, density, scale, rotation);

    return true;
}


/// storing rigid infos needed for RigidMass
struct GenerateRigidInfo
{
    type::Matrix3 inertia;
    type::Quat<SReal> inertia_rotation;
    type::Vec3 inertia_diagonal;
    type::Vec3 com;
    SReal mass;
};

/// user friendly function to compute rigid info from a mesh, a density, a scale
template<typename Rigid3MassType>
void generateRigid( GenerateRigidInfo& res
                                  , io::Mesh *mesh
                                  , std::string const& meshName
                                  , SReal density
                                  , const type::Vec3& scale = type::Vec3(1,1,1)
                                  , const type::Vec3& rotation /*Euler angles*/ = type::Vec3(0,0,0)
                                  )

{
    Rigid3MassType rigidMass;

    generateRigid( rigidMass, res.com, mesh, density, scale, rotation  );

    if( rigidMass.mass < 0 )
    {
        msg_warning("generatedRigid")<<"are normals inverted? "<<meshName;
        rigidMass.mass = -rigidMass.mass;
        rigidMass.inertiaMatrix = -rigidMass.inertiaMatrix;
    }

    res.mass = rigidMass.mass;
    res.inertia = res.mass * rigidMass.inertiaMatrix;

    // a threshold to test if inertia is diagonal in function of diagonal values
    const SReal threshold = type::trace( res.inertia ) * 1e-6;

    // if not diagonal, extracting principal axes basis to get the corresponding rotation with a diagonal inertia
    if( res.inertia[0][1]>threshold || res.inertia[0][2]>threshold || res.inertia[1][2]>threshold )
    {
        type::Matrix3 U;
        Decompose<SReal>::eigenDecomposition_iterative( res.inertia, U, res.inertia_diagonal );

        // det should be 1->rotation or -1->reflexion
        if( type::determinant( U ) < 0 ) // reflexion
        {
            // made it a rotation by negating a column
            U[0][0] = -U[0][0];
            U[1][0] = -U[1][0];
            U[2][0] = -U[2][0];
        }
        res.inertia_rotation.fromMatrix( U );
    }
    else
    {
        res.inertia_diagonal[0] = res.inertia[0][0];
        res.inertia_diagonal[1] = res.inertia[1][1];
        res.inertia_diagonal[2] = res.inertia[2][2];
        res.inertia_rotation.clear();
    }
}

/// user friendly function to compute rigid info from a mesh file, a density, a scale
template<typename Rigid3MassType>
bool generateRigid( GenerateRigidInfo& res
                                  , const std::string& meshFilename
                                  , SReal density
                                  , const type::Vec3& scale = type::Vec3(1,1,1)
                                  , const type::Vec3& rotation /*Euler angles*/ = type::Vec3(0,0,0)
                                  )
{
    sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create( meshFilename );
    if (mesh == nullptr)
    {
        msg_info("GenerateRigid") << "unable to load mesh from file '"<<meshFilename<<"'" ;
        return false;
    }

    generateRigid<Rigid3MassType>(res, mesh, meshFilename, density, scale, rotation);
    return true;
}

} // namespace sofa::helper
