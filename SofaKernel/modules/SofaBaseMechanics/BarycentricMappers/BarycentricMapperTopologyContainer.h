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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_H

#include <SofaBaseTopology/TopologyData.inl>
#include <SofaBaseMechanics/BarycentricMappers/TopologyBarycentricMapper.h>

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _barycentricmappertopologycontainer_
{

using sofa::defaulttype::Mat3x3d;
using sofa::defaulttype::Vector3;


/// Template class for topology container mappers
template<class In, class Out, class MappingDataType, class Element>
class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer : public TopologyBarycentricMapper<In,Out>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef typename Inherit1::Real Real;
    typedef typename Inherit1::OutReal OutReal;
    typedef typename Inherit1::OutDeriv  OutDeriv;
    typedef typename Inherit1::InDeriv  InDeriv;

    typedef typename Inherit1::MBloc MBloc;
    typedef typename Inherit1::MatrixType MatrixType;

    typedef typename Inherit1::ForceMask ForceMask;
    typedef typename MatrixType::Index MatrixTypeIndex;
    enum { NIn = Inherit1::NIn };
    enum { NOut = Inherit1::NOut };

public:

    virtual void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    virtual void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    virtual void clear(int size=0) override;
    virtual void resize( core::State<Out>* toModel ) override;

    virtual void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    virtual void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    virtual void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    virtual void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    virtual const sofa::defaulttype::BaseMatrix* getJ(int outSize, int inSize) override;

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

    bool isEmpty();

protected:

    using Inherit1::m_fromTopology;

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
         : Inherit1(fromTopology, toTopology),
           d_map(initData(&d_map,"map", "mapper data")),
           m_matrixJ(NULL),
           m_updateJ(true)
     {}

    virtual ~BarycentricMapperTopologyContainer() override {}

protected:

    virtual helper::vector<Element> getElements()=0;
    virtual helper::vector<SReal> getBaryCoef(const Real* f)=0;
    virtual void computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Element& element)=0;
    virtual void computeCenter(Vector3& center, const typename In::VecCoord& in, const Element& element)=0;
    virtual void addPointInElement(const int elementIndex, const SReal* baryCoords)=0;
    virtual void computeDistance(double& d, const Vector3& v)=0;

    void exhaustiveSearch ( defaulttype::Vec3d outPos,
                            const typename In::VecCoord& in,
                            const helper::vector<Mat3x3d>& bases,
                            const helper::vector<Vector3>& centers);

    unsigned int getHashIndexFromCoord(const Vector3& x);
    unsigned int getHashIndexFromIndices(const int& x, const int& y, const int& z);
    defaulttype::Vec3i getGridIndices(const Vector3& x);
    void addToHashTable(const unsigned int& hId, const unsigned int& vertexId);
    void initHashing(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void computeHashingCellSize(const typename In::VecCoord& in);
    void computeBB(const typename Out::VecCoord& out, const typename In::VecCoord& in);
    void computeHashTable(const typename In::VecCoord& in);

};


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec3fTypes;
typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_CPP)
#ifndef SOFA_FLOAT
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
#endif
#ifndef SOFA_DOUBLE
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
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
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
#endif
#endif
#endif

}

using _barycentricmappertopologycontainer_::BarycentricMapperTopologyContainer;

}}}

#endif
