#ifndef CPUSPHFLUIDFORCEFIELDWITHOPENCL_CPP
#define CPUSPHFLUIDFORCEFIELDWITHOPENCL_CPP

#include "myopencl.h"
#include "CPUSPHFluidForceField.h"

using namespace sofa::gpu::opencl;

class CPUSPHFluidForceFieldWithOpenCL:public CPUSPHFluidForceField
{
public:
    static void addForce(unsigned int _gsize, const _device_pointer _cells, const _device_pointer _cellGhost, GPUSPHFluid* params,_device_pointer _f, const _device_pointer _pos4, const _device_pointer _v);
};

#endif // CPUSPHFLUIDFORCEFIELDWITHOPENCL_CPP
