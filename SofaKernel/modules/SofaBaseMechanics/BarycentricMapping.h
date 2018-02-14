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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_H
#include "config.h"

#include <SofaBaseTopology/TopologyData.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>

#include <sofa/helper/vector.h>

// forward declarations
namespace sofa
{
namespace core
{
namespace topology
{
class BaseMeshTopology;
}
}

namespace component
{
namespace topology
{
class MeshTopology;
class RegularGridTopology;
class SparseGridTopology;

class PointSetTopologyContainer;
template <class T>
class PointSetGeometryAlgorithms;

class EdgeSetTopologyContainer;
template <class T>
class EdgeSetGeometryAlgorithms;

class TriangleSetTopologyContainer;
template <class T>
class TriangleSetGeometryAlgorithms;

class QuadSetTopologyContainer;
template <class T>
class QuadSetGeometryAlgorithms;

class TetrahedronSetTopologyContainer;
template <class T>
class TetrahedronSetGeometryAlgorithms;

class HexahedronSetTopologyContainer;
template <class T>
class HexahedronSetGeometryAlgorithms;
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
class BarycentricMapper : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapper,In,Out),core::objectmodel::BaseObject);

    typedef typename In::Real Real;
    typedef typename In::Real InReal;
    typedef typename Out::Real OutReal;

    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Deriv InDeriv;

    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Deriv OutDeriv;

    enum { NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size };
    enum { NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size };
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;

protected:

    void addMatrixContrib(MatrixType* m, int row, int col, Real value)
    {
        MBloc* b = m->wbloc(row, col, true); // get write access to a matrix bloc, creating it if not found
        for (int i=0; i < ((int)NIn < (int)NOut ? (int)NIn : (int)NOut); ++i)
            (*b)[i][i] += value;
    }

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
            out << "\n";
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

protected:
    BarycentricMapper() {}
    virtual ~BarycentricMapper() {}

private:
    BarycentricMapper(const BarycentricMapper& n) ;
    BarycentricMapper& operator=(const BarycentricMapper& n) ;

public:
    using core::objectmodel::BaseObject::init;
    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;
    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) = 0;
    virtual const sofa::defaulttype::BaseMatrix* getJ(int /*outSize*/, int /*inSize*/)
    {
        dmsg_error() << " getJ() NOT IMPLEMENTED BY " << sofa::core::objectmodel::BaseClass::decodeClassName(typeid(*this)) ;
        return NULL;
    }
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) = 0;
    virtual void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) = 0;

    using core::objectmodel::BaseObject::draw;
    virtual void draw(const core::visual::VisualParams*, const typename Out::VecCoord& out, const typename In::VecCoord& in) = 0;

    //-- test mapping partiel
    virtual void applyOnePoint( const unsigned int& /*hexaId*/, typename Out::VecCoord& /*out*/, const typename In::VecCoord& /*in*/)
    {}
    //--


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
    SOFA_CLASS(SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out),SOFA_TEMPLATE2(BarycentricMapper,In,Out));

    typedef BarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename core::behavior::BaseMechanicalState::ForceMask ForceMask;

    ForceMask *maskFrom;
    ForceMask *maskTo;
//    core::State<Out>* toModel;

protected:
    virtual ~TopologyBarycentricMapper() {}
public:

    virtual int addPointInLine(const int /*lineIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int setPointInLine(const int /*pointIndex*/, const int /*lineIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInLine(const typename Out::Coord& /*p*/, int /*lineIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInTriangle(const int /*triangleIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int setPointInTriangle(const int /*pointIndex*/, const int /*triangleIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInTriangle(const typename Out::Coord& /*p*/, int /*triangleIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInQuad(const int /*quadIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int setPointInQuad(const int /*pointIndex*/, const int /*quadIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInQuad(const typename Out::Coord& /*p*/, int /*quadIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInTetra(const int /*tetraIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int setPointInTetra(const int /*pointIndex*/, const int /*tetraIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInTetra(const typename Out::Coord& /*p*/, int /*tetraIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual int addPointInCube(const int /*cubeIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int setPointInCube(const int /*pointIndex*/, const int /*cubeIndex*/, const SReal* /*baryCoords*/) {return 0;}
    virtual int createPointInCube(const typename Out::Coord& /*p*/, int /*cubeIndex*/, const typename In::VecCoord* /*points*/) {return 0;}

    virtual void setToTopology( topology::PointSetTopologyContainer* toTopology) {this->toTopology = toTopology;}
    const topology::PointSetTopologyContainer *getToTopology() const {return toTopology;}


    virtual void updateForceMask()
    {
        // mask is already filled in the mapper's applyJT;
    }

    virtual void resize( core::State<Out>* toModel ) = 0;

protected:
    TopologyBarycentricMapper(core::topology::BaseMeshTopology* fromTopology, topology::PointSetTopologyContainer* toTopology = NULL)
        : fromTopology(fromTopology), toTopology(toTopology)
    {}

protected:
    core::topology::BaseMeshTopology* fromTopology;
    topology::PointSetTopologyContainer* toTopology;
};



/// Class allowing barycentric mapping computation on a MeshTopology
template<class In, class Out>
class BarycentricMapperMeshTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperMeshTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));

    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;

    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData1D MappingData1D;
    typedef typename Inherit::MappingData2D MappingData2D;
    typedef typename Inherit::MappingData3D MappingData3D;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    void addMatrixContrib(MatrixType* m, int row, int col, Real value)
    {
        Inherit::addMatrixContrib(m, row, col, value);
    }

    sofa::helper::vector< MappingData1D >  map1d;
    sofa::helper::vector< MappingData2D >  map2d;
    sofa::helper::vector< MappingData3D >  map3d;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, toTopology),
          matrixJ(NULL), updateJ(true)
    {
    }

    virtual ~BarycentricMapperMeshTopology()
    {
        if (matrixJ) delete matrixJ;
    }
public:

    void clear(int reserve=0) override;

    int addPointInLine(const int lineIndex, const SReal* baryCoords) override;
    int createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points) override;

    int addPointInTriangle(const int triangleIndex, const SReal* baryCoords) override;
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points) override;

    int addPointInQuad(const int quadIndex, const SReal* baryCoords) override;
    int createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points) override;

    int addPointInTetra(const int tetraIndex, const SReal* baryCoords) override;

    int addPointInCube(const int cubeIndex, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

    sofa::helper::vector< MappingData3D > const* getMap3d() const { return &map3d; }

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



/// Class allowing barycentric mapping computation on a RegularGridTopology
template<class In, class Out>
class BarycentricMapperRegularGridTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperRegularGridTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::CubeData CubeData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    void addMatrixContrib(MatrixType* m, int row, int col, Real value)
    {
        Inherit::addMatrixContrib(m, row, col, value);
    }

    sofa::helper::vector<CubeData> map;
    topology::RegularGridTopology* fromTopology;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperRegularGridTopology(topology::RegularGridTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),fromTopology(fromTopology),
          matrixJ(NULL), updateJ(true)
    {
    }

    virtual ~BarycentricMapperRegularGridTopology()
    {
        if (matrixJ) delete matrixJ;
    }
public:

    void clear(int reserve=0) override;

    bool isEmpty() {return this->map.size() == 0;}
    void setTopology(topology::RegularGridTopology* _topology) {this->fromTopology = _topology;}
    topology::RegularGridTopology *getTopology() {return dynamic_cast<topology::RegularGridTopology *>(this->fromTopology);}

    int addPointInCube(const int cubeIndex, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

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
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperSparseGridTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;

    typedef typename Inherit::CubeData CubeData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    void addMatrixContrib(MatrixType* m, int row, int col, Real value)
    {
        Inherit::addMatrixContrib(m, row, col, value);
    }

    sofa::helper::vector<CubeData> map;
    topology::SparseGridTopology* fromTopology;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperSparseGridTopology(topology::SparseGridTopology* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          fromTopology(fromTopology),
          matrixJ(NULL), updateJ(true)
    {
    }

    virtual ~BarycentricMapperSparseGridTopology()
    {
        if (matrixJ) delete matrixJ;
    }
public:

    void clear(int reserve=0) override;

    int addPointInCube(const int cubeIndex, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

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

/// Class allowing barycentric mapping computation on a EdgeSetTopology
template<class In, class Out>
class BarycentricMapperEdgeSetTopology : public TopologyBarycentricMapper<In,Out>

{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperEdgeSetTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData1D MappingData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    topology::PointData< sofa::helper::vector<MappingData > > map; ///< mapper data
    topology::EdgeSetTopologyContainer*			_fromContainer;
    topology::EdgeSetGeometryAlgorithms<In>*	_fromGeomAlgo;
    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperEdgeSetTopology(topology::EdgeSetTopologyContainer* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(fromTopology),
          _fromGeomAlgo(NULL),
          matrixJ(NULL),
          updateJ(true)
    {}

    virtual ~BarycentricMapperEdgeSetTopology() {}
public:

    void clear(int reserve=0) override;

    int addPointInLine(const int edgeIndex, const SReal* baryCoords) override;
    int createPointInLine(const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperEdgeSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        sofa::helper::vector<MappingData>& m = *(b.map.beginEdit());
        m.clear();

        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            m.push_back(value);
        }
        b.map.endEdit();
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperEdgeSetTopology<In, Out> & b )
    {

        out << b.map.getValue().size();
        out << " " ;
        out << b.map;

        return out;
    }
};



/// Class allowing barycentric mapping computation on a TriangleSetTopology
template<class In, class Out>
class BarycentricMapperTriangleSetTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTriangleSetTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData2D MappingData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    topology::PointData< sofa::helper::vector<MappingData> > map; ///< mapper data
    topology::TriangleSetTopologyContainer*			_fromContainer;
    topology::TriangleSetGeometryAlgorithms<In>*	_fromGeomAlgo;
    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperTriangleSetTopology(topology::TriangleSetTopologyContainer* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(fromTopology),
          _fromGeomAlgo(NULL),
          matrixJ(NULL),
          updateJ(true)
    {
    }

    virtual ~BarycentricMapperTriangleSetTopology(){}

public:
    void clear(int reserve=0) override;

    virtual int addPointInTriangle(const int triangleIndex, const SReal* baryCoords) override;
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

#ifdef BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
    // handle topology changes in the From topology
    virtual void handleTopologyChange(core::topology::Topology* t);
#endif // BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperTriangleSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;

        sofa::helper::vector<MappingData>& m = *(b.map.beginEdit());
        m.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            m.push_back(value);
        }
        b.map.endEdit();
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperTriangleSetTopology<In, Out> & b )
    {

        out << b.map.getValue().size();
        out << " " ;
        out << b.map;

        return out;
    }
};



/// Class allowing barycentric mapping computation on a QuadSetTopology
template<class In, class Out>
class BarycentricMapperQuadSetTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperQuadSetTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData2D MappingData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    topology::PointData< sofa::helper::vector<MappingData> >  map; ///< mapper data
    topology::QuadSetTopologyContainer*			_fromContainer;
    topology::QuadSetGeometryAlgorithms<In>*	_fromGeomAlgo;
    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperQuadSetTopology(topology::QuadSetTopologyContainer* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(fromTopology),
          _fromGeomAlgo(NULL),
          matrixJ(NULL),
          updateJ(true)
    {}

    virtual ~BarycentricMapperQuadSetTopology() {}

public:
    void clear(int reserve=0) override;

    int addPointInQuad(const int index, const SReal* baryCoords) override;
    int createPointInQuad(const typename Out::Coord& p, int index, const typename In::VecCoord* points) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperQuadSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        sofa::helper::vector<MappingData>& m = *(b.map.beginEdit());
        m.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            m.push_back(value);
        }
        b.map.endEdit();
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperQuadSetTopology<In, Out> & b )
    {

        out << b.map.getValue().size();
        out << " " ;
        out << b.map;

        return out;
    }

};

/// Class allowing barycentric mapping computation on a TetrahedronSetTopology
template<class In, class Out>
class BarycentricMapperTetrahedronSetTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData3D MappingData;

    typedef typename In::VecCoord VecCoord;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    topology::PointData< sofa::helper::vector<MappingData > >  map; ///< mapper data

    VecCoord actualTetraPosition;

    topology::TetrahedronSetTopologyContainer*			_fromContainer;
    topology::TetrahedronSetGeometryAlgorithms<In>*	_fromGeomAlgo;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(fromTopology),
          _fromGeomAlgo(NULL),
          matrixJ(NULL),
          updateJ(true)
    {}

    virtual ~BarycentricMapperTetrahedronSetTopology() {}

public:
    void clear(int reserve=0) override;

    int addPointInTetra(const int index, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;


};



/// Class allowing barycentric mapping computation on a HexahedronSetTopology
template<class In, class Out>
class BarycentricMapperHexahedronSetTopology : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapperHexahedronSetTopology,In,Out),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;
    typedef typename Inherit::MappingData3D MappingData;

    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };
    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;
    typedef typename MatrixType::Index MatrixTypeIndex;

    typedef typename Inherit::ForceMask ForceMask;

protected:
    topology::PointData< sofa::helper::vector<MappingData> >  map; ///< mapper data
    topology::HexahedronSetTopologyContainer*		_fromContainer;
    topology::HexahedronSetGeometryAlgorithms<In>*	_fromGeomAlgo;

    std::set<int>	_invalidIndex;
    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperHexahedronSetTopology()
        : TopologyBarycentricMapper<In,Out>(NULL, NULL),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(NULL),_fromGeomAlgo(NULL)
    {}

    BarycentricMapperHexahedronSetTopology(topology::HexahedronSetTopologyContainer* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          map(initData(&map,"map", "mapper data")),
          _fromContainer(fromTopology),
          _fromGeomAlgo(NULL),
          matrixJ(NULL),
          updateJ(true)
    {}

    virtual ~BarycentricMapperHexahedronSetTopology() {}

public:
    void clear(int reserve=0) override;

    int addPointInCube(const int index, const SReal* baryCoords) override;

    int setPointInCube(const int pointIndex, const int cubeIndex, const SReal* baryCoords) override;

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void resize( core::State<Out>* toModel ) override;

    //-- test mapping partiel
    void applyOnePoint( const unsigned int& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    //--

    // handle topology changes in the From topology
    virtual void handleTopologyChange(core::topology::Topology* t) override;

    bool isEmpty() {return this->map.getValue().empty();}
    void setTopology(topology::HexahedronSetTopologyContainer* _topology) {this->fromTopology = _topology; _fromContainer=_topology;}
    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperHexahedronSetTopology<In, Out> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        sofa::helper::vector<MappingData>& m = *(b.map.beginEdit());
        m.clear();
        MappingData value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            m.push_back(value);
        }
        b.map.endEdit();
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperHexahedronSetTopology<In, Out> & b )
    {

        out << b.map.getValue().size();
        out << " " ;
        out << b.map;

        return out;
    }


};



template <class TIn, class TOut>
class BarycentricMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BarycentricMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;
    typedef Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::Real OutReal;

    typedef core::topology::BaseMeshTopology BaseMeshTopology;

    typedef TopologyBarycentricMapper<InDataTypes,OutDataTypes> Mapper;
    //typedef BarycentricMapperRegularGridTopology<InDataTypes, OutDataTypes> RegularGridMapper;
    //typedef BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes> HexaMapper;

    typedef typename Inherit::ForceMask ForceMask;

protected:

    SingleLink<BarycentricMapping<In,Out>,Mapper,BaseLink::FLAG_STRONGLINK> mapper;

public:

    Data< bool > useRestPosition; ///< Use the rest position of the input and output models to initialize the mapping

#ifdef SOFA_DEV
    //--- partial mapping test
    Data< bool > sleeping; ///< is the mapping sleeping (not computed)
#endif
protected:
    BarycentricMapping();

    BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr mapper);

    BarycentricMapping(core::State<In>* from, core::State<Out>* to, BaseMeshTopology * topology=NULL );

    virtual ~BarycentricMapping();

public:
    void init() override;

    void reinit() override;

    void apply(const core::MechanicalParams *mparams, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data< typename Out::VecDeriv >& out, const Data< typename In::VecDeriv >& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in) override;

    virtual const sofa::defaulttype::BaseMatrix* getJ() override;


public:
    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs() override;

protected:
    typedef linearsolver::EigenSparseMatrix<InDataTypes, OutDataTypes> eigen_type;

    // eigen matrix for use with Compliant plugin
    eigen_type eigen;
    helper::vector< defaulttype::BaseMatrix* > js;

    virtual void updateForceMask() override;

public:

    void draw(const core::visual::VisualParams* vparams) override;

    // handle topology changes depending on the topology
    virtual void handleTopologyChange(core::topology::Topology* t) override;

    // interface for continuous friction contact



    TopologyBarycentricMapper<InDataTypes,OutDataTypes> *getMapper()
    {
        return mapper.get();
    }

protected:
    sofa::core::topology::BaseMeshTopology* topology_from;
    sofa::core::topology::BaseMeshTopology* topology_to;

private:
    void createMapperFromTopology(BaseMeshTopology * topology);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
