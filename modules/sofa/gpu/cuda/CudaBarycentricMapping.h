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

template <typename VecIn, typename VecOut>
class BarycentricMapperRegularGridTopology< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> > : public TopologyBarycentricMapper< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<VecIn,VecIn,float> In;
    typedef gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> Out;
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::CubeData CubeData;
protected:
    gpu::cuda::CudaVector<CubeData> map;
    int maxNOut;
    gpu::cuda::CudaVector< std::pair<int,float> > mapT;
    topology::RegularGridTopology* topology;
    void calcMapT();
public:
    BarycentricMapperRegularGridTopology(topology::RegularGridTopology* topology)
        : TopologyBarycentricMapper(topology),
          maxNOut(0), topology(topology)
    {}

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperRegularGridTopology<In, Out> &b )
    {
        in >> b.map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperRegularGridTopology<In, Out> & b )
    {
        out << b.map;
        return out;
    }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
