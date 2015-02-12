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
#include "GenerateRigid.h"

#include <stdlib.h>
#include <math.h>
#include <sofa/helper/vector.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace helper
{

using namespace sofa::defaulttype;

void GenerateRigid(Rigid3Mass& mass, Vector3& center, const sofa::helper::io::Mesh* mesh)
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
    double afIntegral[10] = { (double)0.0, (double)0.0, (double)0.0, (double)0.0,
            (double)0.0, (double)0.0, (double)0.0, (double)0.0, (double)0.0, (double)0.0
                            };

    const vector<Vector3>& points = mesh->getVertices();
    const vector< vector< vector<int> > >& facets = mesh->getFacets();
    for (unsigned int i = 0; i < facets.size(); i++)
    {
        const vector<int>& v = facets[i][0];
        for (unsigned int j = 2; j < v.size(); j++)
        {
            // get vertices of current triangle
            const Vector3 kV0 = points[v[0  ]];
            const Vector3 kV1 = points[v[j-1]];
            const Vector3 kV2 = points[v[j  ]];

            // get cross product of edges
            Vector3 kV1mV0 = kV1 - kV0;
            Vector3 kV2mV0 = kV2 - kV0;
            Vector3 kN = cross(kV1mV0,kV2mV0);

            // compute integral terms
            double fTmp0, fTmp1, fTmp2;
            double fF1x, fF2x, fF3x, fG0x, fG1x, fG2x;
            fTmp0 = kV0[0] + kV1[0];
            fF1x = fTmp0 + kV2[0];
            fTmp1 = kV0[0]*kV0[0];
            fTmp2 = fTmp1 + kV1[0]*fTmp0;
            fF2x = fTmp2 + kV2[0]*fF1x;
            fF3x = kV0[0]*fTmp1 + kV1[0]*fTmp2 + kV2[0]*fF2x;
            fG0x = fF2x + kV0[0]*(fF1x + kV0[0]);
            fG1x = fF2x + kV1[0]*(fF1x + kV1[0]);
            fG2x = fF2x + kV2[0]*(fF1x + kV2[0]);

            double fF1y, fF2y, fF3y, fG0y, fG1y, fG2y;
            fTmp0 = kV0[1] + kV1[1];
            fF1y = fTmp0 + kV2[1];
            fTmp1 = kV0[1]*kV0[1];
            fTmp2 = fTmp1 + kV1[1]*fTmp0;
            fF2y = fTmp2 + kV2[1]*fF1y;
            fF3y = kV0[1]*fTmp1 + kV1[1]*fTmp2 + kV2[1]*fF2y;
            fG0y = fF2y + kV0[1]*(fF1y + kV0[1]);
            fG1y = fF2y + kV1[1]*(fF1y + kV1[1]);
            fG2y = fF2y + kV2[1]*(fF1y + kV2[1]);

            double fF1z, fF2z, fF3z, fG0z, fG1z, fG2z;
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

    afIntegral[0] /= 6.0;
    afIntegral[1] /= 24.0;
    afIntegral[2] /= 24.0;
    afIntegral[3] /= 24.0;
    afIntegral[4] /= 60.0;
    afIntegral[5] /= 60.0;
    afIntegral[6] /= 60.0;
    afIntegral[7] /= 120.0;
    afIntegral[8] /= 120.0;
    afIntegral[9] /= 120.0;

    // mass
    mass.volume = afIntegral[0];
    mass.mass = mass.volume;

    // center of mass
    center = Vector3(afIntegral[1]/afIntegral[0],afIntegral[2]/afIntegral[0],afIntegral[3]/afIntegral[0]);

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

    mass.recalc();
}



bool GenerateRigid(Rigid3Mass& mass, Vector3& center, const std::string& meshFilename
                   , SReal density
                   , const Vector3& scale
                   , const Vector3& rotation /*Euler angles*/
                  )
{
    sofa::helper::io::Mesh* mesh = sofa::helper::io::Mesh::Create( meshFilename );
    if (mesh == NULL)
    {
        std::cout << "ERROR loading mesh "<<meshFilename<<std::endl;
        return false;
    }

    if( scale != Vector3(1, 1, 1) ) {
        for(unsigned i = 0, n = mesh->getVertices().size(); i < n; ++i) {
            mesh->getVertices()[i] = mesh->getVertices()[i].linearProduct(scale);
        }
    }

    if( rotation != Vector3(0,0,0) ) {

        Quaternion q = sofa::helper::Quater<SReal>::createQuaterFromEuler( rotation*M_PI/180.0 );

        for(unsigned i = 0, n = mesh->getVertices().size(); i < n; ++i) {
            mesh->getVertices()[i] = q.rotate( mesh->getVertices()[i] );
        }
    }

    GenerateRigid( mass, center, mesh );

    mass.mass *= density;

    return true;
}

}

}
