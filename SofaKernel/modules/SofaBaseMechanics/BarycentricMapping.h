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

#include <sofa/core/topology/BaseMeshTopology.h>

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


    virtual void clearMapAndReserve( int reserve=0 ) =0;

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

    virtual void updateForceMask(){/*mask is already filled in the mapper's applyJT*/}

    virtual void resize( core::State<Out>* toModel ) = 0;

protected:
    core::topology::BaseMeshTopology* m_fromTopology;
    topology::PointSetTopologyContainer* toTopology;

    TopologyBarycentricMapper(core::topology::BaseMeshTopology* fromTopology, topology::PointSetTopologyContainer* toTopology = NULL)
        : m_fromTopology(fromTopology)
        , toTopology(toTopology)
    {}

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

    sofa::helper::vector< MappingData1D >  m_map1d;
    sofa::helper::vector< MappingData2D >  m_map2d;
    sofa::helper::vector< MappingData3D >  m_map3d;

    MatrixType* m_matrixJ;
    bool m_updateJ;

    BarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, toTopology),
          m_matrixJ(NULL), m_updateJ(true)
    {
    }

    virtual ~BarycentricMapperMeshTopology()
    {
        if (m_matrixJ) delete m_matrixJ;
    }
public:

    void clearMapAndReserve(int reserve=0) override;

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

    sofa::helper::vector< MappingData3D > const* getMap3d() const { return &m_map3d; }

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperMeshTopology<In, Out> &b )
    {
        unsigned int size_vec;
        in >> size_vec;
        b.m_map1d.clear();
        MappingData1D value1d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value1d;
            b.m_map1d.push_back(value1d);
        }

        in >> size_vec;
        b.m_map2d.clear();
        MappingData2D value2d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value2d;
            b.m_map2d.push_back(value2d);
        }

        in >> size_vec;
        b.m_map3d.clear();
        MappingData3D value3d;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value3d;
            b.m_map3d.push_back(value3d);
        }
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperMeshTopology<In, Out> & b )
    {

        out << b.m_map1d.size();
        out << " " ;
        out << b.m_map1d;
        out << " " ;
        out << b.m_map2d.size();
        out << " " ;
        out << b.m_map2d;
        out << " " ;
        out << b.m_map3d.size();
        out << " " ;
        out << b.m_map3d;

        return out;
    }

private:
    void clearMap1dAndReserve(int size=0);
    void clearMap2dAndReserve(int size=0);
    void clearMap3dAndReserve(int size=0);

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

    sofa::helper::vector<CubeData> m_map;
    topology::RegularGridTopology* m_fromTopology;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperRegularGridTopology(topology::RegularGridTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),m_fromTopology(fromTopology),
          matrixJ(NULL), updateJ(true)
    {
    }

    virtual ~BarycentricMapperRegularGridTopology()
    {
        if (matrixJ) delete matrixJ;
    }
public:

    void clearMapAndReserve(int reserve=0) override;

    bool isEmpty() {return this->m_map.size() == 0;}
    void setTopology(topology::RegularGridTopology* _topology) {this->m_fromTopology = _topology;}
    topology::RegularGridTopology *getTopology() {return dynamic_cast<topology::RegularGridTopology *>(this->m_fromTopology);}

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
        in >> b.m_map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperRegularGridTopology<In, Out> & b )
    {
        out << b.m_map;
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

    sofa::helper::vector<CubeData> m_map;
    topology::SparseGridTopology* m_fromTopology;

    MatrixType* matrixJ;
    bool updateJ;

    BarycentricMapperSparseGridTopology(topology::SparseGridTopology* fromTopology,
            topology::PointSetTopologyContainer* _toTopology)
        : TopologyBarycentricMapper<In,Out>(fromTopology, _toTopology),
          m_fromTopology(fromTopology),
          matrixJ(NULL), updateJ(true)
    {
    }

    virtual ~BarycentricMapperSparseGridTopology()
    {
        if (matrixJ) delete matrixJ;
    }
public:

    void clearMapAndReserve(int reserve=0) override;

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
        in >> b.m_map;
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperSparseGridTopology<In, Out> & b )
    {
        out << b.m_map;
        return out;
    }

};






typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;
using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;
using sofa::defaulttype::Vec3i;


/// Template class for topology container mappers
template<class In, class Out, class MappingDataType, class Element>
class BarycentricMapperTopologyContainer : public TopologyBarycentricMapper<In,Out>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef TopologyBarycentricMapper<In,Out> Inherit;
    typedef typename Inherit::Real Real;
    typedef typename Inherit::OutReal OutReal;
    typedef typename Inherit::OutDeriv  OutDeriv;
    typedef typename Inherit::InDeriv  InDeriv;

    typedef typename Inherit::MBloc MBloc;
    typedef typename Inherit::MatrixType MatrixType;

    typedef typename Inherit::ForceMask ForceMask;
    typedef typename MatrixType::Index MatrixTypeIndex;
    enum { NIn = Inherit::NIn };
    enum { NOut = Inherit::NOut };

protected:

    using Inherit::m_fromTopology;

    topology::PointData< helper::vector<MappingDataType > > d_map;
    MatrixType* m_matrixJ;
    bool m_updateJ;

    // Spacial hashing utils
    Real m_gridCellSize;
    Real m_convFactor;
    unsigned int m_hashTableSize;
    helper::vector<helper::vector<unsigned int>> m_hashTable;
    bool m_computeDistances;

    BarycentricMapperTopologyContainer(core::topology::BaseMeshTopology* fromTopology, topology::PointSetTopologyContainer* toTopology)
         : Inherit(fromTopology, toTopology),
           d_map(initData(&d_map,"map", "mapper data")),
           m_matrixJ(NULL),
           m_updateJ(true)
     {}

    virtual ~BarycentricMapperTopologyContainer()
    {
        if (m_matrixJ)
            delete m_matrixJ;
    }

    unsigned int getHashIndexFromCoord(const Vector3& x)
    {
        Vec3i v = getGridIndices(x);
        return getHashIndexFromIndices(v[0],v[1],v[2]);
    }

    unsigned int getHashIndexFromIndices(const int& x, const int& y, const int& z)
    {
        unsigned int h = (73856093*x^19349663*y^83492791*z)%m_hashTableSize;
        return h;
    }

    Vec3i getGridIndices(const Vector3& x)
    {
        Vec3i i_x;
        for(int i=0; i<3; i++)
            i_x[i]=floor(x[i]*m_convFactor);

        return i_x;
    }

    void addToHashTable(const unsigned int& hId, const unsigned int& vertexId)
    {
        if(hId<m_hashTableSize)
            m_hashTable[hId].push_back(vertexId);
    }

protected:

    virtual helper::vector<Element> getElements()=0;
    virtual helper::vector<Real> getBaryCoef(const Real* f)=0;
    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element)=0;
    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element)=0;
    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords)=0;
    virtual void computeDistance(double& d, const Vector3& v)=0;

    void initHashing(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void computeHashingCellSize(const typename In::VecCoord& in);
    void computeBB(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void computeHashTable(const typename In::VecCoord& in);

public:

    virtual void clearMapAndReserve(int size=0) override;
    virtual void resize( core::State<Out>* toModel ) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;
    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    inline friend std::istream& operator >> ( std::istream& in, BarycentricMapperTopologyContainer<In, Out, MappingDataType, Element> &b )
    {
        unsigned int size_vec;

        in >> size_vec;
        sofa::helper::vector<MappingDataType>& m = *(b.d_map.beginEdit());
        m.clear();

        MappingDataType value;
        for (unsigned int i=0; i<size_vec; i++)
        {
            in >> value;
            m.push_back(value);
        }
        b.d_map.endEdit();
        return in;
    }

    inline friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperTopologyContainer<In, Out, MappingDataType, Element> & b )
    {

        out << b.d_map.getValue().size();
        out << " " ;
        out << b.d_map;

        return out;
    }

    bool isEmpty()
    {
        return d_map.getValue().empty();
    }

};



/////// Class allowing barycentric mapping computation on a EdgeSetTopology
template<class In, class Out, class MappingDataType = typename BarycentricMapper<In,Out>::MappingData1D, class Element = Edge>
class BarycentricMapperEdgeSetTopology : public BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperEdgeSetTopology,In,Out,MappingDataType,Element),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element));
    typedef MappingDataType MappingData;
    typedef BarycentricMapperTopologyContainer<In,Out,MappingData,Element> Inherit;
    typedef typename Inherit::Real Real;

protected:
    topology::EdgeSetTopologyContainer*	m_fromContainer;
    topology::EdgeSetGeometryAlgorithms<In>* m_fromGeomAlgo;

    using Inherit::d_map;
    using Inherit::m_fromTopology;
    using Inherit::m_matrixJ;
    using Inherit::m_updateJ;

    BarycentricMapperEdgeSetTopology()
         : Inherit(NULL,NULL),
           m_fromContainer(NULL),
           m_fromGeomAlgo(NULL)
    {}

    BarycentricMapperEdgeSetTopology(topology::EdgeSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperEdgeSetTopology() {}

    virtual helper::vector<Element> getElements() override
    {
        return this->m_fromTopology->getEdges();
    }

    virtual helper::vector<Real> getBaryCoef(const Real* f) override
    {
        return getBaryCoef(f[0]);
    }

    helper::vector<Real> getBaryCoef(const Real fx)
    {
        helper::vector<Real> edgeCoef{1-fx,fx};
        return edgeCoef;
    }

    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element) override
    {
        //Not implemented for Edge
        SOFA_UNUSED(base);
        SOFA_UNUSED(in);
        SOFA_UNUSED(element);
    }

    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element) override
    {
        center = (in[element[0]]+in[element[1]])*0.5;
    }

    virtual void computeDistance(double& d, const Vector3& v) override
    {
        //Not implemented for Edge
        SOFA_UNUSED(d);
        SOFA_UNUSED(v);
    }

    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override
    {
        addPointInLine(elementIndex,baryCoords);
    }

public:

    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override
    {
        SOFA_UNUSED(out);
        SOFA_UNUSED(in);
        msg_warning() << "Mapping not implemented for edge elements.";
    }
    virtual int addPointInLine(const int edgeIndex, const SReal* baryCoords) override;
    virtual int createPointInLine(const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points) override;
};



/// Class allowing barycentric mapping computation on a TriangleSetTopology
template<class In, class Out, class MappingDataType = typename BarycentricMapper<In,Out>::MappingData2D, class Element = Triangle>
class BarycentricMapperTriangleSetTopology : public BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperTriangleSetTopology,In,Out,MappingDataType,Element),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element));
    typedef MappingDataType MappingData;
    typedef BarycentricMapperTopologyContainer<In,Out,MappingData,Element> Inherit;
    typedef typename Inherit::Real Real;

protected:
    topology::TriangleSetTopologyContainer*			m_fromContainer;
    topology::TriangleSetGeometryAlgorithms<In>*	m_fromGeomAlgo;

    using Inherit::d_map;
    using Inherit::m_fromTopology;
    using Inherit::m_matrixJ;
    using Inherit::m_updateJ;

    BarycentricMapperTriangleSetTopology()
         : Inherit(NULL,NULL),
           m_fromContainer(NULL),
           m_fromGeomAlgo(NULL)
    {}

    BarycentricMapperTriangleSetTopology(topology::TriangleSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperTriangleSetTopology() {}

    virtual helper::vector<Element> getElements() override
    {
        return this->m_fromTopology->getTriangles();
    }

    virtual helper::vector<Real> getBaryCoef(const Real* f) override
    {
        return getBaryCoef(f[0],f[1]);
    }

    helper::vector<Real> getBaryCoef(const Real fx, const Real fy)
    {
        helper::vector<Real> triangleCoef{1-fx-fy, fx, fy};
        return triangleCoef;
    }

    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element) override
    {
        Mat3x3d mt;
        base[0] = in[element[1]]-in[element[0]];
        base[1] = in[element[2]]-in[element[0]];
        base[2] = cross(base[0],base[1]);
        mt.transpose(base);
        base.invert(mt);
    }

    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element) override
    {
        center = (in[element[0]]+in[element[1]]+in[element[2]])/3;
    }

    virtual void computeDistance(double& d, const Vector3& v) override
    {
        d = std::max ( std::max ( -v[0],-v[1] ),std::max ( ( v[2]<0?-v[2]:v[2] )-0.01,v[0]+v[1]-1 ) );
    }

    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override
    {
        addPointInTriangle(elementIndex,baryCoords);
    }

public:

    virtual int addPointInTriangle(const int triangleIndex, const SReal* baryCoords) override;
    int createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points) override;

#ifdef BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
    // handle topology changes in the From topology
    virtual void handleTopologyChange(core::topology::Topology* t);
#endif // BARYCENTRIC_MAPPER_TOPOCHANGE_REINIT
};



/// Class allowing barycentric mapping computation on a QuadSetTopology
template<class In, class Out, class MappingDataType = typename BarycentricMapper<In,Out>::MappingData2D, class Element = Quad>
class BarycentricMapperQuadSetTopology : public BarycentricMapperTopologyContainer<In,Out,MappingDataType, Element>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperQuadSetTopology,In,Out,MappingDataType,Element),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element));
    typedef MappingDataType MappingData;
    typedef BarycentricMapperTopologyContainer<In,Out,MappingData,Element> Inherit;
    typedef typename Inherit::Real Real;

protected:
    topology::QuadSetTopologyContainer*			m_fromContainer;
    topology::QuadSetGeometryAlgorithms<In>*	m_fromGeomAlgo;

    using Inherit::d_map;
    using Inherit::m_fromTopology;
    using Inherit::m_matrixJ;
    using Inherit::m_updateJ;

    BarycentricMapperQuadSetTopology(topology::QuadSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperQuadSetTopology() {}

    virtual helper::vector<Element> getElements() override
    {
        return this->m_fromTopology->getQuads();
    }

    virtual helper::vector<Real> getBaryCoef(const Real* f) override
    {
        return getBaryCoef(f[0],f[1]);
    }

    helper::vector<Real> getBaryCoef(const Real fx, const Real fy)
    {
        helper::vector<Real> quadCoef{(1-fx)*(1-fy),
                    (fx)*(1-fy),
                    (1-fx)*(fy),
                    (fx)*(fy)};
        return quadCoef;
    }

    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element) override
    {
        Mat3x3d matrixTranspose;
        base[0] = in[element[1]]-in[element[0]];
        base[1] = in[element[3]]-in[element[0]];
        base[2] = cross(base[0],base[1]);
        matrixTranspose.transpose(base);
        base.invert(matrixTranspose);
    }

    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element) override
    {
        center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]] ) *0.25;
    }

    virtual void computeDistance(double& d, const Vector3& v) override
    {
        d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
    }

    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override
    {
        addPointInQuad(elementIndex,baryCoords);
    }

public:

    virtual int addPointInQuad(const int index, const SReal* baryCoords) override;
    virtual int createPointInQuad(const typename Out::Coord& p, int index, const typename In::VecCoord* points) override;
};



/// Class allowing barycentric mapping computation on a TetrahedronSetTopology
template<class In, class Out, class MappingDataType = typename BarycentricMapper<In,Out>::MappingData3D, class Element = Tetrahedron>
class BarycentricMapperTetrahedronSetTopology : public BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperTetrahedronSetTopology,In,Out,MappingDataType,Element),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element));
    typedef MappingDataType MappingData;
    typedef BarycentricMapperTopologyContainer<In,Out,MappingData,Element> Inherit;
    typedef typename Inherit::Real Real;

protected:

    topology::TetrahedronSetTopologyContainer*      m_fromContainer;
    topology::TetrahedronSetGeometryAlgorithms<In>*	m_fromGeomAlgo;

    using Inherit::d_map;
    using Inherit::m_matrixJ;
    using Inherit::m_updateJ;
    using Inherit::m_fromTopology;

    BarycentricMapperTetrahedronSetTopology()
        : Inherit(NULL,NULL),
          m_fromContainer(NULL),
          m_fromGeomAlgo(NULL)
    {}

    BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperTetrahedronSetTopology() {}

    virtual helper::vector<Element> getElements() override
    {
        return this->m_fromTopology->getTetrahedra();
    }

    virtual helper::vector<Real> getBaryCoef(const Real* f) override
    {
        return getBaryCoef(f[0],f[1],f[2]);
    }

    helper::vector<Real> getBaryCoef(const Real fx, const Real fy, const Real fz)
    {
        helper::vector<Real> tetrahedronCoef{(1-fx-fy-fz),fx,fy,fz};
        return tetrahedronCoef;
    }

    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element) override
    {
        Mat3x3d matrixTranspose;
        base[0] = in[element[1]]-in[element[0]];
        base[1] = in[element[2]]-in[element[0]];
        base[2] = in[element[3]]-in[element[0]];
        matrixTranspose.transpose(base);
        base.invert(matrixTranspose);
    }

    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element) override
    {
        center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]] ) *0.25;
    }

    virtual void computeDistance(double& d, const Vector3& v) override
    {
        d = std::max ( std::max ( -v[0],-v[1] ), std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
    }

    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override
    {
        addPointInTetra(elementIndex,baryCoords);
    }

public:

    virtual int addPointInTetra(const int index, const SReal* baryCoords) override ;
};



/// Class allowing barycentric mapping computation on a HexahedronSetTopology
template<class In, class Out, class MappingDataType = typename BarycentricMapper<In, Out>::MappingData3D, class Element = Hexahedron>
class BarycentricMapperHexahedronSetTopology : public BarycentricMapperTopologyContainer<In,Out,MappingDataType,Element>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperHexahedronSetTopology,In,Out,MappingDataType,Element),SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element));
    typedef MappingDataType MappingData;
    typedef BarycentricMapperTopologyContainer<In,Out,MappingData,Element> Inherit;
    typedef typename Inherit::Real Real;

protected:
    topology::HexahedronSetTopologyContainer*		m_fromContainer;
    topology::HexahedronSetGeometryAlgorithms<In>*	m_fromGeomAlgo;
    std::set<int> m_invalidIndex;

    using Inherit::d_map;
    using Inherit::m_matrixJ;
    using Inherit::m_updateJ;
    using Inherit::m_fromTopology;

    BarycentricMapperHexahedronSetTopology()
        : Inherit(NULL, NULL),
          m_fromContainer(NULL),
          m_fromGeomAlgo(NULL)
    {}

    BarycentricMapperHexahedronSetTopology(topology::HexahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
        : Inherit(fromTopology, toTopology),
          m_fromContainer(fromTopology),
          m_fromGeomAlgo(NULL)
    {}

    virtual ~BarycentricMapperHexahedronSetTopology() {}


    virtual helper::vector<Element> getElements() override
    {
        return this->m_fromTopology->getHexahedra();
    }

    virtual helper::vector<Real> getBaryCoef(const Real* f) override
    {
        return getBaryCoef(f[0],f[1],f[2]);
    }

    helper::vector<Real> getBaryCoef(const Real fx, const Real fy, const Real fz)
    {
        helper::vector<Real> hexahedronCoef{(1-fx)*(1-fy)*(1-fz),
                    (fx)*(1-fy)*(1-fz),
                    (1-fx)*(fy)*(1-fz),
                    (fx)*(fy)*(1-fz),
                    (1-fx)*(1-fy)*(fz),
                    (fx)*(1-fy)*(fz),
                    (1-fx)*(fy)*(fz),
                    (fx)*(fy)*(fz)};
        return hexahedronCoef;
    }

    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element) override
    {
        Mat3x3d matrixTranspose;
        base[0] = in[element[1]]-in[element[0]];
        base[1] = in[element[3]]-in[element[0]];
        base[2] = in[element[4]]-in[element[0]];
        matrixTranspose.transpose(base);
        base.invert(matrixTranspose);
    }

    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element) override
    {
        center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]]+in[element[4]]+in[element[5]]+in[element[6]]+in[element[7]] ) *0.125;
    }

    virtual void computeDistance(double& d, const Vector3& v) override
    {
        d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( -v[2],v[0]-1 ),std::max ( v[1]-1,v[2]-1 ) ) );
    }

    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords) override
    {
        addPointInCube(elementIndex,baryCoords);
    }

public:

    virtual int addPointInCube(const int index, const SReal* baryCoords) override;
    virtual int setPointInCube(const int pointIndex, const int cubeIndex, const SReal* baryCoords) override;
    //-- test mapping partiel
    virtual void applyOnePoint( const unsigned int& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    //--
    // handle topology changes in the From topology
    virtual void handleTopologyChange(core::topology::Topology* t) override;

    void setTopology(topology::HexahedronSetTopologyContainer* topology)
    {
        m_fromTopology  = topology;
        m_fromContainer = topology;
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
    typedef Out OutDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef typename OutDataTypes::Real OutReal;

    typedef core::topology::BaseMeshTopology BaseMeshTopology;
    typedef TopologyBarycentricMapper<InDataTypes,OutDataTypes> Mapper;
    typedef typename Inherit::ForceMask ForceMask;

protected:

    SingleLink<BarycentricMapping<In,Out>,Mapper,BaseLink::FLAG_STRONGLINK> m_mapper;

public:

    Data< bool > useRestPosition; ///< Use the rest position of the input and output models to initialize the mapping

#ifdef SOFA_DEV
    //--- partial mapping test
    Data< bool > sleeping; ///< is the mapping sleeping (not computed)
#endif
protected:
    BarycentricMapping();

    BarycentricMapping(core::State<In>* from, core::State<Out>* to, typename Mapper::SPtr m_mapper);

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
        return m_mapper.get();
    }

protected:
    sofa::core::topology::BaseMeshTopology* topology_from;
    sofa::core::topology::BaseMeshTopology* topology_to;

private:
    void createMapperFromTopology(BaseMeshTopology * topology);
};


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData1D, Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData1D, Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData2D, Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData2D, Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData2D, Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData2D, Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData3D, Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData3D, Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData3D, Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData3D, Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3dTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, ExtVec3fTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData1D, Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData1D, Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData2D, Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData2D, Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData2D, Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData2D, Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData3D, Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData3D, Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData3D, Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData3D, Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3fTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, ExtVec3fTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData1D,Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData1D,Edge>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData2D,Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData2D,Triangle>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData2D,Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData2D,Quad>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData3D,Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData3D,Tetrahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData3D,Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData3D,Hexahedron>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3fTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3dTypes>;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
