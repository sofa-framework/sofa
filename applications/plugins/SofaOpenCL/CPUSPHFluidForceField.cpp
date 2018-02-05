/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "CPUSPHFluidForceField.h"


CPUSPHFluidForceField::float3 CPUSPHFluidForceField::SPHFluidCalcForce(CPUSPHFluidForceField::float4 x1, CPUSPHFluidForceField::float3 v1, CPUSPHFluidForceField::float4 x2, CPUSPHFluidForceField::float3 v2, CPUSPHFluidForceField::float3 force, GPUSPHFluid params)
{
    float3 n(x2.x()-x1.x(),x2.y()-x1.y(),x2.z()-x1.z());
    float d2 = n.x()*n.x()+n.y()*n.y()+n.z()*n.z();
    if (d2 < params.h2)
    {
        //float r_h = rsqrtf(params->h2/d2);
        //float r_h = sqrtf(d2/params->h2);
        float r2_h2 = (d2/params.h2);
        float r_h = sqrt(r2_h2);
        float density1 = x1.w();
        float density2 = x2.w();

        // Pressure
        float pressure1 = params.stiffness * (density1 - params.density0);
        float pressure2 = params.stiffness * (density2 - params.density0);
        float3 pressure = gradWp(n, r_h, params.CgradWp) * ( params.mass2 * (pressure1 / (density1*density1) + pressure2 / (density2*density2) ));
        force += pressure;
        // Viscosity
        float3 viscosity = (( v2 - v1 ) * (params.mass2 * params.viscosity * laplacianWv(r_h,params.ClaplacianWv) / (density1 * density2) ) );
        force += viscosity;

    }
    return force;
}

void CPUSPHFluidForceField::vectorAddForce(unsigned int gsize, const int*cells, const int*cellGhost,GPUSPHFluid params,CPUSPHFluidForceField::float3* f, const CPUSPHFluidForceField::float4 *pos4, const CPUSPHFluidForceField::float3* v)
{
    for(unsigned int cell = 0; cell<gsize; cell++)
    {
        int range_x = cells[cell];
        int range_y = cells[cell+1] & ~(1U<<31);
        int ghost = cellGhost[cell];

        if (range_x <=0) continue;

        for(int px = range_x; px < ghost; px++)
        {
            float3 force(0,0,0);
            for(int py = range_x; py < range_y; py++)
            {
                if(px!=py)
                {
                    force = SPHFluidCalcForce(pos4[cells[px]], v[cells[px]], pos4[cells[py]], v[cells[py]], force, params);
                }
            }
            f[cells[px]]+=force;
        }
    }
}
