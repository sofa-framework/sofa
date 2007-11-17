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
class TopologyBarycentricMapper<topology::RegularGridTopology,gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes> : public BarycentricMapper<gpu::cuda::CudaVec3fTypes,gpu::cuda::CudaVec3fTypes>
{
public:
    typedef gpu::cuda::CudaVec3fTypes In;
    typedef gpu::cuda::CudaVec3fTypes Out;
    typedef BarycentricMapper<In,Out> Inherit;
    typedef Inherit::Real Real;
    typedef Inherit::OutReal OutReal;
protected:
    gpu::cuda::CudaVector<CubeData> map;
    int maxNOut;
    gpu::cuda::CudaVector< std::pair<int,float> > mapT;
    topology::RegularGridTopology* topology;
public:
    TopologyBarycentricMapper(topology::RegularGridTopology* topology) : maxNOut(0), topology(topology)
    {}

    bool empty() const {return map.size()==0;}

    void setTopology( topology::RegularGridTopology* t ) { topology = t; }

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    void init();

    void apply( Out::VecCoord& out, const In::VecCoord& in );
    void applyJ( Out::VecDeriv& out, const In::VecDeriv& in );
    void applyJT( In::VecDeriv& out, const Out::VecDeriv& in );
    void applyJT( In::VecConst& out, const Out::VecConst& in );
    void draw( const Out::VecCoord& out, const In::VecCoord& in);

    inline friend std::istream& operator >> ( std::istream& in, TopologyBarycentricMapper<topology::RegularGridTopology, In, Out> &b )
    {
        in >> b.map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const TopologyBarycentricMapper<topology::RegularGridTopology, In, Out> & b )
    {
        out << b.map;
        return out;
    }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
