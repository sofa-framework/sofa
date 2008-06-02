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
        : Inherit(topology)
        , maxNOut(0), topology(topology)
    {}

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const Real* baryCoords);

    bool isEmpty() { return map.size() == 0; }
    void setTopology(topology::RegularGridTopology* _topology) { topology = _topology; }

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

template <typename VecIn, typename VecOut>
class BarycentricMapperMeshTopology< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> > : public TopologyBarycentricMapper< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<VecIn,VecIn,float> In;
    typedef gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> Out;
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    class MapData
    {
    public:
        class GPULinearData
        {
        public:
            int i;
            float val;
        };
        GPULinearData d[BSIZE];
        //int i[BSIZE];
        //float val[BSIZE];
    };
protected:
    int maxNIn;
    int maxNOut;
    int insize,size;
    gpu::cuda::CudaVector< MapData > map;
    gpu::cuda::CudaVector< MapData > mapT;
    core::componentmodel::topology::BaseMeshTopology* topology;
    void resizeMap(int size2, int maxNIn2);
    void setMap(int outIndex, int j, int inIndex, Real val);
    void calcMapT();
public:
    BarycentricMapperMeshTopology(core::componentmodel::topology::BaseMeshTopology* topology)
        : Inherit(topology)
        , maxNIn(0), maxNOut(0), insize(0), size(0), topology(topology)
    {
        if (topology==NULL || topology->getNbHexas()==0) maxNIn = 4;
        else maxNIn = 8;
    }

    void clear(int reserve=0);

    int addPointInLine(int lineIndex, const SReal* baryCoords);
    int createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points);

    int addPointInTriangle(int triangleIndex, const SReal* baryCoords);
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points);

    int addPointInQuad(int quadIndex, const SReal* baryCoords);
    int createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points);

    int addPointInTetra(int tetraIndex, const SReal* baryCoords);

    int addPointInCube(int cubeIndex, const SReal* baryCoords);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperMeshTopology<In, Out> &b )
    {
        in >> b.map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperMeshTopology<In, Out> & b )
    {
        out << b.map;
        return out;
    }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
