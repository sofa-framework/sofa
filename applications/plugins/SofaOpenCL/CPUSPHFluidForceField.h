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
#ifndef SOFAOPENCL_CPUSPHFLUIDFORCEFIELD_H
#define SOFAOPENCL_CPUSPHFLUIDFORCEFIELD_H

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <sofa/defaulttype/Vec.h>


class CPUSPHFluidForceField
{
public:
    enum {NUM_ELEMENTS=1000};

    typedef sofa::defaulttype::Vec4f float4 ;
    typedef sofa::defaulttype::Vec3f float3 ;

    typedef struct
    {
        float h;         ///< particles radius
        float h2;        ///< particles radius squared
        float stiffness; ///< pressure stiffness
        float mass;      ///< particles mass
        float mass2;     ///< particles mass squared
        float density0;  ///< 1000 kg/m3 for water
        float viscosity;
        float surfaceTension;

        // Precomputed constants for smoothing kernels
        float CWd;          ///< =     constWd(h)
        float CgradWd;      ///< = constGradWd(h)
        float CgradWp;      ///< = constGradWp(h)
        float ClaplacianWv; ///< =  constLaplacianWv(h)
        float CgradWc;      ///< = constGradWc(h)
        float ClaplacianWc; ///< =  constLaplacianWc(h)
    } GPUSPHFluid;

    static float3 gradWp(const float3 d, float r_h, float C)
    {
        float a = 1-r_h;
        return d*C*a*a;
    }

    static float laplacianWv(float r_h, float C)
    {
        return C*(1-r_h);
    }

    static float3 SPHFluidCalcForce(float4 x1, float3 v1, float4 x2, float3 v2, float3 force, GPUSPHFluid params);

    static void vectorAddForce(unsigned int gsize, const int*cells, const int*cellGhost,GPUSPHFluid params,float3* f, const float4 *pos4, const float3* v);

};



#endif // CPUSPHFLUIDFORCEFIELD_H
