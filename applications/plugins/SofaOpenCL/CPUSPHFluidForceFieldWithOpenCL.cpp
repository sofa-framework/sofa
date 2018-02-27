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
#include "CPUSPHFluidForceFieldWithOpenCL.h"


void CPUSPHFluidForceFieldWithOpenCL::addForce(unsigned int _gsize, const _device_pointer _cells, const _device_pointer _cellGhost, GPUSPHFluid* params,_device_pointer _f, const _device_pointer _pos4, const _device_pointer _v)
{
    float3 *f = new float3[NUM_ELEMENTS];
    float4 *pos4 = new float4[NUM_ELEMENTS];
    int *cells = new int[32800+8*NUM_ELEMENTS];
    int *cellGhost = new int[32768];
    float3 *v = new float3[NUM_ELEMENTS];

    sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,f,_f.m,_f.offset,sizeof(float3)*NUM_ELEMENTS);
    sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,pos4,_pos4.m,_pos4.offset,sizeof(float4)*NUM_ELEMENTS);
    sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,cells,_cells.m,_cells.offset,sizeof(int)*(32800+8*NUM_ELEMENTS));
    sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,cellGhost,_cellGhost.m,_cellGhost.offset,sizeof(int)*32768);
    sofa::gpu::opencl::myopenclEnqueueReadBuffer(0,v,_v.m,_v.offset,sizeof(float3)*NUM_ELEMENTS);

    vectorAddForce(_gsize,cells,cellGhost,*params,f,pos4,v);

    sofa::gpu::opencl::myopenclEnqueueWriteBuffer(0,_v.m,_v.offset,v,sizeof(float3)*NUM_ELEMENTS);
    sofa::gpu::opencl::myopenclEnqueueWriteBuffer(0,_f.m,_f.offset,f,sizeof(float3)*NUM_ELEMENTS);

    delete [] pos4;
    delete [] cells;
    delete [] cellGhost;
    delete [] v;
    delete [] f;
}
