#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_INL
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_INL

#include "CudaBarycentricMapping.h"
#include <sofa/component/mapping/BarycentricMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJT(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;


void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::clear(int reserve)
{
    map.clear();
    if (reserve>0) map.reserve(reserve);
}

int RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::addPointInCube(int cubeIndex, const Real* baryCoords)
{
    map.resize(map.size()+1);
    CubeData& data = map[map.size()-1];
    //data.in_index = cubeIndex;
    data.in_index = topology->getCube(cubeIndex)[0];
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    return map.size()-1;
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::init()
{
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_apply(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_applyJ(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };
    RegularGridMapperCuda3f_applyJT(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecConst& out, const Out::VecConst& in )
{
}

void RegularGridMapper<CudaVec3fTypes,CudaVec3fTypes>::draw( const Out::VecCoord& out, const In::VecCoord& in)
{
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
