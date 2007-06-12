#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_H
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
class RegularGridMapper<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes> : public BarycentricMapper<gpu::cuda::CudaVec3fTypes,gpu::cuda::CudaVec3fTypes>
{
public:
    typedef gpu::cuda::CudaVec3fTypes In;
    typedef gpu::cuda::CudaVec3fTypes Out;
    typedef BarycentricMapper<In,Out> Inherit;
    typedef Inherit::Real Real;
    typedef Inherit::OutReal OutReal;
protected:
    gpu::cuda::CudaVector<CubeData> map;
    topology::RegularGridTopology* topology;
public:
    RegularGridMapper(topology::RegularGridTopology* topology) : topology(topology)
    {}

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    void init();

    void apply( Out::VecCoord& out, const In::VecCoord& in );
    void applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
    void applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
    void applyJT( In::VecConst& out, const Out::VecConst& in );
    void draw( const Out::VecCoord& out, const In::VecCoord& in);
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
