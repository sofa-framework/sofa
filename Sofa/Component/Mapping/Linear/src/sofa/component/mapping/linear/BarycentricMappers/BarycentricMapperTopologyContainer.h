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
#pragma once
#include <sofa/component/mapping/linear/BarycentricMappers/TopologyBarycentricMapper.h>

#include <sofa/core/topology/TopologyData.inl>
#include <unordered_map>

namespace sofa::component::mapping::linear::_barycentricmappertopologycontainer_
{

using sofa::type::Mat3x3;
using sofa::type::Vec3;
using sofa::type::Vec3i;
using sofa::defaulttype::Vec3Types;
typedef typename sofa::core::topology::BaseMeshTopology::Edge Edge;
typedef typename sofa::core::topology::BaseMeshTopology::Triangle Triangle;
typedef typename sofa::core::topology::BaseMeshTopology::Quad Quad;
typedef typename sofa::core::topology::BaseMeshTopology::Tetrahedron Tetrahedron;
typedef typename sofa::core::topology::BaseMeshTopology::Hexahedron Hexahedron;

/// Template class for topology container mappers
template<class In, class Out, class MappingDataType, class Element>
class BarycentricMapperTopologyContainer : public TopologyBarycentricMapper<In,Out>
{

public:

    SOFA_CLASS(SOFA_TEMPLATE4(BarycentricMapperTopologyContainer,In,Out,MappingDataType,Element),SOFA_TEMPLATE2(TopologyBarycentricMapper,In,Out));
    typedef typename Inherit1::Real Real;
    typedef typename Inherit1::OutReal OutReal;
    typedef typename Inherit1::OutDeriv  OutDeriv;
    typedef typename Inherit1::InDeriv  InDeriv;

    typedef typename Inherit1::MBloc MBloc;
    typedef typename Inherit1::MatrixType MatrixType;

    typedef typename MatrixType::Index MatrixTypeIndex;
    enum { NIn = Inherit1::NIn };
    enum { NOut = Inherit1::NOut };

    using Index = sofa::Index;

public:

    void init(const typename Out::VecCoord& out, const typename In::VecCoord& in) override;
    void draw(const core::visual::VisualParams*,const typename Out::VecCoord& out, const typename In::VecCoord& in) override;

    void clear(std::size_t size=0) override;
    void resize( core::State<Out>* toModel ) override;

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in ) override;
    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in ) override;
    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in ) override;
    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) override;
    const sofa::linearalgebra::BaseMatrix* getJ(int outSize, int inSize) override;

    template<class I, class O, class MDType, class E>
    friend std::istream& operator >> ( std::istream& in, BarycentricMapperTopologyContainer<I, O, MDType, E> &b );
    template<class I, class O, class MDType, class E>
    friend std::ostream& operator << ( std::ostream& out, const BarycentricMapperTopologyContainer<I, O, MDType, E> & b );

    bool isEmpty();

protected:

    struct Key
    {
        Key(const int& xId, const int& yId, const int& zId)
        {
            this->xId=xId;
            this->yId=yId;
            this->zId=zId;
        }

        int xId,yId,zId; // cell indices
    };

    struct HashFunction
    {
        size_t operator()(const Key &key) const
        {
            // We use the large prime numbers proposed in paper:
            // M.Teschner et al "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (2003)
            const int h = (73856093*key.xId^19349663*key.yId^83492791*key.zId);
            return size_t(h);
        }
    };

    struct HashEqual
    {
        bool operator()(const Key &key1, const Key &key2) const
        {
            return ((key1.xId==key2.xId) && (key1.yId==key2.yId) && (key1.zId==key2.zId));
        }
    };

    struct NearestParams
    {
        NearestParams()
        {
            distance = std::numeric_limits<SReal>::max();
            elementId = std::numeric_limits<unsigned int>::max();
        }

        sofa::type::Vec3 baryCoords;
        SReal distance;
        unsigned int elementId;
    };

    using Inherit1::m_fromTopology;

    core::topology::PointData< type::vector<MappingDataType > > d_map; ///< mapper data
    MatrixType* m_matrixJ {nullptr};
    bool m_updateJ {false};

    type::vector<Mat3x3> m_bases;
    type::vector<Vec3> m_centers;

    // Spacial hashing utils
    Real m_gridCellSize;
    Real m_convFactor;
    std::unordered_map<Key, type::vector<unsigned int>, HashFunction, HashEqual> m_hashTable;
    std::size_t m_hashTableSize;


    BarycentricMapperTopologyContainer(sofa::core::topology::TopologyContainer* fromTopology, core::topology::BaseMeshTopology* toTopology);

    ~BarycentricMapperTopologyContainer() override = default;

    virtual type::vector<Element> getElements()=0;
    virtual std::array<Real, Element::NumberOfNodes>getBarycentricCoefficients(const std::array<Real, MappingDataType::NumberOfCoordinates>& barycentricCoordinates)=0;
    
    virtual void computeBase(Mat3x3& base, const typename In::VecCoord& in, const Element& element)=0;
    virtual void computeCenter(Vec3& center, const typename In::VecCoord& in, const Element& element)=0;
    virtual void addPointInElement(const Index elementIndex, const SReal* baryCoords)=0;
    virtual void computeDistance(SReal& d, const Vec3& v)=0;

    /// Compute the distance between outPos and the element e. If this distance is smaller than the previously stored one,
    /// update nearestParams.
    /// \param e id of the element
    /// \param outPos position of the point we want to compute the barycentric coordinates
    /// \param inPos position of one point of the element
    /// \param nearestParams output parameters (nearest element id, distance, and barycentric coordinates)
    void checkDistanceFromElement(unsigned int e,
                                  const Vec3& outPos,
                                  const Vec3& inPos,
                                  NearestParams& nearestParams);


    /// Compute the datas needed to find the nearest element
    /// \param in is the vector of points
    void computeBasesAndCenters( const typename In::VecCoord& in );

    // Spacial hashing following paper:
    // M.Teschner et al "Optimized Spatial Hashing for Collision Detection of Deformable Objects" (2003)
    type::Vec3i getGridIndices(const Vec3& pos);
    void initHashing(const typename In::VecCoord& in);
    void computeHashingCellSize(const typename In::VecCoord& in);
    void computeHashTable(const typename In::VecCoord& in);

};

#if !defined(SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData1D, Edge>;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData2D, Triangle>;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData2D, Quad>;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData3D, Tetrahedron>;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData3D, Hexahedron>;


#endif

} // namespace sofa::component::mapping::linear::_barycentricmappertopologycontainer_

namespace sofa::component::mapping::linear
{

using _barycentricmappertopologycontainer_::BarycentricMapperTopologyContainer;

} // namespace sofa::component::mapping::linear
