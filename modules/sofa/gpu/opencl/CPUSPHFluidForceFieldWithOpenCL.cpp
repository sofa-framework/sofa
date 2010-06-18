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

    delete(pos4);
    delete(cells);
    delete(cellGhost);
    delete(v);
    delete(f);
}
