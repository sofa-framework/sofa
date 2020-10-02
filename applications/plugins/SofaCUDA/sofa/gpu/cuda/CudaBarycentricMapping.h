/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_H
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_H

#include "CudaTypes.h"
#include <SofaBaseMechanics/BarycentricMapping.h>
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperRegularGridTopology.h>
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperSparseGridTopology.h>
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <SofaBaseMechanics/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h>
#include <sofa/core/behavior/MechanicalState.h>

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
    using index_type = sofa::index_type;

protected:
    gpu::cuda::CudaVector<CubeData> map;
    std::size_t maxNOut;
    gpu::cuda::CudaVector< std::pair<int,float> > mapT;
    topology::RegularGridTopology* topology;
    void calcMapT();
public:
    BarycentricMapperRegularGridTopology(topology::RegularGridTopology* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology)
        , maxNOut(0), topology(fromTopology)
    {}
    void setMaskFrom(helper::StateMask *) {}
    void setMaskTo  (helper::StateMask *) {}

    void clear(std::size_t reserve=0);

    index_type addPointInCube(const index_type cubeIndex, const SReal* baryCoords);

    bool isEmpty() { return map.size() == 0; }
    void setTopology(topology::RegularGridTopology* _topology) { topology = _topology; }

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void resize( core::State<Out>* toModel );

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
class BarycentricMapperSparseGridTopology< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> > : public TopologyBarycentricMapper< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<VecIn,VecIn,float> In;
    typedef gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> Out;
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::CubeData CubeData;
    using index_type = sofa::index_type;
protected:
    gpu::cuda::CudaVector<CubeData> map;
    topology::SparseGridTopology* topology;
    bool bHexa;
    bool bTrans;
    unsigned sizeout;
    gpu::cuda::CudaVector<unsigned int> CudaHexa;
    gpu::cuda::CudaVector<unsigned int> CudaTid;
    gpu::cuda::CudaVector<unsigned int> CudaTnb;
    gpu::cuda::CudaVector<unsigned int> CudaTst;
    gpu::cuda::CudaVector<float> CudaTVal;
    void buildHexa();
    void buildTranslate(unsigned outsize);

public:
    BarycentricMapperSparseGridTopology(topology::SparseGridTopology* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology)
        , topology(fromTopology), bHexa(true), bTrans(true)
    {}
    void setMaskFrom(helper::StateMask *) {}
    void setMaskTo  (helper::StateMask *) {}

    void clear(std::size_t reserve=0);

    index_type addPointInCube(const index_type cubeIndex, const SReal* baryCoords);

    bool isEmpty() { return map.size() == 0; }
    void setTopology(topology::RegularGridTopology* _topology) { topology = _topology; }

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void resize( core::State<Out>* toModel );

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperSparseGridTopology<In, Out> &b )
    {
        in >> b.map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperSparseGridTopology<In, Out> & b )
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
    using index_type = sofa::index_type;

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
    std::size_t maxNIn;
    std::size_t maxNOut;
    std::size_t insize,size;
    gpu::cuda::CudaVector< MapData > map;
    gpu::cuda::CudaVector< MapData > mapT;
    core::topology::BaseMeshTopology* topology;
    void resizeMap(std::size_t size2, std::size_t maxNIn2);
    void setMap(index_type outIndex, index_type j, index_type inIndex, Real val);
    float getMapValue(index_type outIndex, index_type j);
    index_type getMapIndex(index_type outIndex, index_type j);
    void calcMapT();
public:
    BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology)
        , maxNIn(0), maxNOut(0), insize(0), size(0), topology(fromTopology)
    {
        if (topology==NULL || topology->getNbHexahedra()==0) maxNIn = 4;
        else maxNIn = 8;
    }
    void setMaskFrom(helper::StateMask *) {}
    void setMaskTo  (helper::StateMask *) {}

    void clear(std::size_t reserve=0);

    index_type addPointInLine(const index_type lineIndex, const SReal* baryCoords);
    index_type createPointInLine(const typename Out::Coord& p, index_type lineIndex, const typename In::VecCoord* points);

    index_type addPointInTriangle(const index_type triangleIndex, const SReal* baryCoords);
    index_type createPointInTriangle(const typename Out::Coord& p, index_type triangleIndex, const typename In::VecCoord* points);

    index_type addPointInQuad(const index_type quadIndex, const SReal* baryCoords);
    index_type createPointInQuad(const typename Out::Coord& p, index_type quadIndex, const typename In::VecCoord* points);

    index_type addPointInTetra(const index_type tetraIndex, const SReal* baryCoords);

    index_type addPointInCube(const index_type cubeIndex, const SReal* baryCoords);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void resize( core::State<Out>* toModel );

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


/// Class allowing barycentric mapping computation on a TetrahedronSetTopology
template<class VecIn, class VecOut>
class BarycentricMapperTetrahedronSetTopology< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> > : public TopologyBarycentricMapper< gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<VecIn,VecIn,float> In;
    typedef gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> Out;
    typedef TopologyBarycentricMapper<In,Out> Inherit;

    typedef typename Inherit::Real Real;
    typedef typename In::VecCoord VecCoord;

    BarycentricMapperMeshTopology< In, Out > internalMapper;

    using index_type = sofa::index_type;

public:
    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* _toTopology)
        : Inherit(fromTopology, _toTopology),
          internalMapper(fromTopology,_toTopology)
    {}

    virtual ~BarycentricMapperTetrahedronSetTopology() {}

    void clear(std::size_t reserve=0) {
        internalMapper.clear(reserve);
    }

    index_type addPointInTetra(const index_type index, const SReal* baryCoords) {
        return internalMapper.addPointInTetra(index,baryCoords);
    }

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) {
        internalMapper.init(out,in);
    }

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) {
        internalMapper.apply(out,in);
    }

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) {
        internalMapper.applyJ(out,in);
    }

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) {
        internalMapper.applyJT(out,in);
    }

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) {
        internalMapper.applyJT(out,in);
    }

    void draw(const core::visual::VisualParams* vp,const typename Out::VecCoord& out, const typename In::VecCoord& in) {
        internalMapper.draw(vp,out,in);
    }

    void resize( core::State<Out>* toModel ) {
        internalMapper.resize(toModel);
    }
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
