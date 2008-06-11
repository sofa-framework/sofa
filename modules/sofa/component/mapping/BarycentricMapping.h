/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H

#include <sofa/defaulttype/Mat.h>
#include <sofa/helper/vector.h>
#include <sofa/component/topology/PointData.h>
#include <sofa/component/topology/HexahedronData.h>

// forward declarations
namespace sofa
{
namespace core
{
namespace componentmodel
{
namespace topology
{
class BaseTopology;
}
}
}

namespace component
{
namespace topology
{
class MeshTopology;
class RegularGridTopology;
class SparseGridTopology;

template<class T>
class EdgeSetTopology;
template<class T>
class TriangleSetTopology;
template<class T>
class QuadSetTopology;
template<class T>
class TetrahedronSetTopology;
template<class T>
class HexahedronSetTopology;
}
}
}

namespace sofa
{

namespace component
{

namespace mapping
{

/// Base class for barycentric mapping topology-specific mappers
template<class In, class Out>
class BarycentricMapper
{
public:
    typedef typename In::Real Real;
    typedef typename Out::Real OutReal;

protected:
    template< int NC,  int NP>
    class MappingData
    {
    public:
        int in_index;
        //unsigned int points[NP];
        Real baryCoords[NC];

        inline friend std::istream& operator >> ( std::istream& in, MappingData< NC, NP> &m )
        {
            in>>m.in_index;
            for (int i=0; i<NC; i++) in >> m.baryCoords[i];
            return in;
        }

        inline friend std::ostream& operator << ( std::ostream& out, const MappingData< NC , NP > & m )
        {
            out << m.in_index;
            for (int i=0; i<NC; i++)
                out << " " << m.baryCoords[i];
            return out;
        }

    };

public:
    typedef MappingData<1,2> LineData;
    typedef MappingData<2,3> TriangleData;
    typedef MappingData<2,4> QuadData;
    typedef MappingData<3,4> TetraData;
    typedef MappingData<3,8> CubeData;
    typedef MappingData<1,0> MappingData1D;
    typedef MappingData<2,0> MappingData2D;
    typedef MappingData<3,0> MappingData3D;

    virtual ~BarycentricMapper() {}
    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;
    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecConst& out, const typename Out::VecConst& in ) = 0;
    virtual void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    virtual void clear( int reserve=0 ) =0;

    //Nothing to do
    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapper< In, Out > & ) {return in;}
    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapper< In, Out > &  ) { return out; }
};

/// Template class for barycentric mapping topology-specific mappers.
template<class In, class Out>
class TopologyBarycentricMapper : public BarycentricMapper<In,Out>
{
public:
    typedef BarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;

    virtual ~TopologyBarycentricMapper() {}

    virtual int addPointInLine(int /*lineIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInLine(const typename Out::Coord& /*p*/, int /*lineIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInTriangle(int /*triangleIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInTriangle(const typename Out::Coord& /*p*/, int /*triangleIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInQuad(int /*quadIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInQuad(const typename Out::Coord& /*p*/, int /*quadIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInTetra(int /*tetraIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInTetra(const typename Out::Coord& /*p*/, int /*tetraIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInCube(int /*cubeIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInCube(const typename Out::Coord& /*p*/, int /*cubeIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

protected:
    TopologyBarycentricMapper(core::componentmodel::topology::Topology* /*topology*/) {}
};

/// Class allowing barycentric mapping computation on a RegularGridTopology
template<class In, class Out>
class BarycentricMapperRegularGridTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::CubeData CubeData;
protected:
    sofa::helper::vector<CubeData> map;
    topology::RegularGridTopology* topology;
public:
    BarycentricMapperRegularGridTopology(topology::RegularGridTopology* topology)
        : TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperRegularGridTopology() {}

    void clear(int reserve=0);
    bool isEmpty() {return map.size() == 0;}
    void setTopology(topology::RegularGridTopology* _topology) {topology = _topology;}
    int addPointInCube(int cubeIndex, const SReal* baryCoords);

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

/// Class allowing barycentric mapping computation on a SparseGridTopology
template<class In, class Out>
class BarycentricMapperSparseGridTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::CubeData CubeData;
protected:
    sofa::helper::vector<CubeData> map;
    topology::SparseGridTopology* topology;
public:
    BarycentricMapperSparseGridTopology(topology::SparseGridTopology* topology)
        : TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperSparseGridTopology() {}

    void clear(int reserve=0);

    int addPointInCube(int cubeIndex, const SReal* baryCoords);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

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


/// Class allowing barycentric mapping computation on a MeshTopology
template<class In, class Out>
class BarycentricMapperMeshTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData1D MappingData1D;
    typedef typename Inherit::MappingData2D MappingData2D;
    typedef typename Inherit::MappingData3D MappingData3D;
protected:
    sofa::helper::vector< MappingData1D >  map1d;
    sofa::helper::vector< MappingData2D >  map2d;
    sofa::helper::vector< MappingData3D >  map3d;
    topology::BaseMeshTopology* topology;

public:
    BarycentricMapperMeshTopology(topology::BaseMeshTopology* topology)
        : TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperMeshTopology() {}

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
        unsigned int size_vec;
        in >> size_vec;
        b.map1d.clear();
        MappingData1D value1d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value1d;
            b.map1d.push_back(value1d);
        }

        in >> size_vec;
        b.map2d.clear();
        MappingData2D value2d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value2d;
            b.map2d.push_back(value2d);
        }

        in >> size_vec;
        b.map3d.clear();
        MappingData3D value3d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value3d;
            b.map3d.push_back(value3d);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperMeshTopology<In, Out> & b )
    {

        out << b.map1d.size();
        out << " " ;
        out << b.map1d;
        out << " " ;
        out << b.map2d.size();
        out << " " ;
        out << b.map2d;
        out << " " ;
        out << b.map3d.size();
        out << " " ;
        out << b.map3d;

        return out;
    }

private:
    void clear1d(int reserve=0);
    void clear2d(int reserve=0);
    void clear3d(int reserve=0);

};

/// Template class for barycentric mapping topology-specific mappers. Enables topological changes.
class BarycentricMapperBaseTopology
{
public:
    virtual ~BarycentricMapperBaseTopology() {}

    // handle topology changes in the From topology
    virtual void handleTopologyChange()=0;
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator )=0;
protected:
    BarycentricMapperBaseTopology(core::componentmodel::topology::BaseTopology* /*topology*/)
    {}
};


/// Class allowing barycentric mapping computation on a EdgeSetTopology
template<class In, class Out>
class BarycentricMapperEdgeSetTopology : public BarycentricMapperBaseTopology, public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData1D MappingData;
protected:
    topology::PointData< MappingData >  map;
    topology::EdgeSetTopology<In>* topology;

public:
    BarycentricMapperEdgeSetTopology(topology::EdgeSetTopology<In>* topology)
        : BarycentricMapperBaseTopology(topology), TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperEdgeSetTopology() {}

    void clear(int reserve=0);

    int addPointInLine(int edgeIndex, const SReal* baryCoords);
    int createPointInLine(const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    // handle topology changes in the From topology
    virtual void handleTopologyChange();
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperEdgeSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        b.map.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            b.map.push_back(value);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperEdgeSetTopology<In, Out> & b )
    {

        out << b.map.size();
        out << " " ;
        out << b.map;

        return out;
    }


};


/// Class allowing barycentric mapping computation on a TriangleSetTopology
template<class In, class Out>
class BarycentricMapperTriangleSetTopology : public BarycentricMapperBaseTopology, public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData2D MappingData;
protected:
    topology::PointData< MappingData >  map;
    topology::TriangleSetTopology<In>* topology;

public:
    BarycentricMapperTriangleSetTopology(topology::TriangleSetTopology<In>* topology)
        : BarycentricMapperBaseTopology(topology), TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperTriangleSetTopology() {}

    void clear(int reserve=0);

    int addPointInTriangle(int triangleIndex, const SReal* baryCoords);
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    // handle topology changes in the From topology
    virtual void handleTopologyChange();
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperTriangleSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        b.map.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            b.map.push_back(value);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperTriangleSetTopology<In, Out> & b )
    {

        out << b.map.size();
        out << " " ;
        out << b.map;

        return out;
    }


};


/// Class allowing barycentric mapping computation on a QuadSetTopology
template<class In, class Out>
class BarycentricMapperQuadSetTopology : public BarycentricMapperBaseTopology, public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData2D MappingData;
protected:
    topology::PointData< MappingData >  map;
    topology::QuadSetTopology<In>* topology;

public:
    BarycentricMapperQuadSetTopology(topology::QuadSetTopology<In>* topology)
        : BarycentricMapperBaseTopology(topology), TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperQuadSetTopology() {}

    void clear(int reserve=0);

    int addPointInQuad(int index, const SReal* baryCoords);
    int createPointInQuad(const typename Out::Coord& p, int index, const typename In::VecCoord* points);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    // handle topology changes in the From topology
    virtual void handleTopologyChange();
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperQuadSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        b.map.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            b.map.push_back(value);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperQuadSetTopology<In, Out> & b )
    {

        out << b.map.size();
        out << " " ;
        out << b.map;

        return out;
    }

};

/// Class allowing barycentric mapping computation on a TetrehedronSetTopology
template<class In, class Out>
class BarycentricMapperTetrahedronSetTopology : public BarycentricMapperBaseTopology, public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData3D MappingData;
protected:
    topology::PointData< MappingData >  map;
    topology::TetrahedronSetTopology<In>* topology;

public:
    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopology<In>* topology)
        : BarycentricMapperBaseTopology(topology), TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperTetrahedronSetTopology() {}

    void clear(int reserve=0);

    int addPointInTetra(int index, const SReal* baryCoords);
//		  int createPointInTetra(const typename Out::Coord& p, int index, const typename In::VecCoord* points);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    // handle topology changes in the From topology
    virtual void handleTopologyChange();
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperTetrahedronSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        b.map.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            b.map.push_back(value);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperTetrahedronSetTopology<In, Out> & b )
    {

        out << b.map.size();
        out << " " ;
        out << b.map;

        return out;
    }
};


/// Class allowing barycentric mapping computation on a HexahedronSetTopology
template<class In, class Out>
class BarycentricMapperHexahedronSetTopology : public BarycentricMapperBaseTopology, public TopologyBarycentricMapper<In,Out>
{
public:
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::MappingData3D MappingData;
protected:
    topology::PointData< MappingData >  map;
    topology::HexahedronSetTopology<In>* topology;

public:
    BarycentricMapperHexahedronSetTopology(topology::HexahedronSetTopology<In>* topology)
        : BarycentricMapperBaseTopology(topology), TopologyBarycentricMapper<In,Out>(topology),
          topology(topology)
    {}

    virtual ~BarycentricMapperHexahedronSetTopology() {}

    void clear(int reserve=0);

    int addPointInCube(int index, const SReal* baryCoords);
//		  int createPointInCube(const typename Out::Coord& p, int index, const typename In::VecCoord* points);

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in);

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );
    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );
    void draw( const typename Out::VecCoord& out, const typename In::VecCoord& in);

    // handle topology changes in the From topology
    virtual void handleTopologyChange();
    // handle topology changes in the To topology
    virtual void handlePointEvents(std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator,
            std::list< const core::componentmodel::topology::TopologyChange *>::const_iterator);

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperHexahedronSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        b.map.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            b.map.push_back(value);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperHexahedronSetTopology<In, Out> & b )
    {

        out << b.map.size();
        out << " " ;
        out << b.map;

        return out;
    }


private:
    struct _BaseAndCenter
    {
        Vector3	origin;
        Matrix3	base;
        Vector3	center;
    } ;

    topology::HexahedronData< _BaseAndCenter > hexahedronData;
};

template <class BasicMapping>
class BarycentricMapping : public BasicMapping
{
public:
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename In::DataTypes InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::SparseVecDeriv InSparseVecDeriv;
    typedef typename InDataTypes::SparseDeriv InSparseDeriv;
    typedef typename InDataTypes::Real Real;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::SparseVecDeriv OutSparseVecDeriv;
    typedef typename OutDataTypes::SparseDeriv OutSparseDeriv;
    typedef typename OutDataTypes::Real OutReal;

    typedef core::componentmodel::topology::Topology Topology;

protected:

    typedef TopologyBarycentricMapper<InDataTypes,OutDataTypes> Mapper;
    typedef BarycentricMapperRegularGridTopology<InDataTypes, OutDataTypes> RegularGridMapper;

    Mapper* mapper;
    DataPtr<  RegularGridMapper >* f_grid;
public:
    BarycentricMapping(In* from, Out* to)
        : Inherit(from, to), mapper(NULL)
        , f_grid (new DataPtr< RegularGridMapper >( new RegularGridMapper( NULL ),"Regular Grid Mapping"))
    {
        this->addField( f_grid, "gridmap");	f_grid->beginEdit();
    }

    BarycentricMapping(In* from, Out* to, Mapper* mapper)
        : Inherit(from, to), mapper(mapper)
    {
        if (RegularGridMapper* m = dynamic_cast< RegularGridMapper* >(mapper))
        {
            f_grid = new DataPtr< RegularGridMapper >( m,"Regular Grid Mapping");
            this->addField( f_grid, "gridmap");	f_grid->beginEdit();
        }
    }

    BarycentricMapping(In* from, Out* to, Topology * topology );

    virtual ~BarycentricMapping()
    {
        if (mapper!=NULL)
            delete mapper;
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    void draw();

    // handle topological changes
    virtual void handleTopologyChange();

    TopologyBarycentricMapper<InDataTypes,OutDataTypes>*	getMapper() {return mapper;}

private:
    void createMapperFromTopology(Topology * topology);
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
