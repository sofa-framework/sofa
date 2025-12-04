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
#ifndef SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_H
#define SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_H

#include <PersistentContact/config.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.h>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperSparseGridTopology.h>
#include <sofa/component/mapping/linear/BarycentricMapping.h>
#include <sofa/component/mapping/linear/BarycentricMappingRigid.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/grid/SparseGridTopology.h>

#include "PersistentContactMapping.h"

namespace sofa
{

namespace component
{

namespace mapping
{

template<class TIn, class TOut>
class PersistentContactBarycentricMapper  : public virtual core::objectmodel::BaseObject // TopologyBarycentricMapper<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut), core::objectmodel::BaseObject);

    typedef TIn In;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecDeriv InVecDeriv;

    /**
     * @brief Add a new contact point in the mapper associated to a persistent contact barycentric mapping.
     */
    virtual int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::type::Vec3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/) = 0;

    /**
     * @brief Add a previously existed contact point in the mapper associated to a persistent contact barycentric mapping.
     */
    virtual int keepContactPointFromInputMapping(const int /*index*/) {return 0;};

    /**
     * @brief Stores a copy of the barycentric data.
     */
    virtual void storeBarycentricData() {};
};



template<class TIn, class TOut>
class PersistentContactBarycentricMapperMeshTopology : public linear::BarycentricMapperMeshTopology<TIn, TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperMeshTopology, TIn, TOut), SOFA_TEMPLATE2(linear::BarycentricMapperMeshTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    PersistentContactBarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
            sofa::component::topology::container::dynamic::PointSetTopologyContainer* toTopology)
        : linear::BarycentricMapperMeshTopology<TIn, TOut>(fromTopology, toTopology)
    {
    }

    virtual ~PersistentContactBarycentricMapperMeshTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::type::Vec3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);
};



template<class TIn, class TOut>
class PersistentContactBarycentricMapperSparseGridTopology : public linear::BarycentricMapperSparseGridTopology<TIn, TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperSparseGridTopology, TIn, TOut), SOFA_TEMPLATE2(linear::BarycentricMapperSparseGridTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    typedef linear::BarycentricMapperSparseGridTopology< TIn , TOut> Inherit;
    typedef typename Inherit::CubeData CubeData;

    PersistentContactBarycentricMapperSparseGridTopology(sofa::component::topology::container::grid::SparseGridTopology* fromTopology,
        topology::container::dynamic::PointSetTopologyContainer* toTopology)
        : linear::BarycentricMapperSparseGridTopology<TIn, TOut>(fromTopology, toTopology)
    {
    }

    virtual ~PersistentContactBarycentricMapperSparseGridTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::type::Vec3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);

    int keepContactPointFromInputMapping(const int /*index*/);

    void storeBarycentricData();

protected:
    sofa::type::vector< CubeData > m_storedMap;
};



template<class TIn, class TOut>
class PersistentContactBarycentricMapperTetrahedronSetTopology : public linear::BarycentricMapperTetrahedronSetTopology<TIn, TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperTetrahedronSetTopology, TIn, TOut), SOFA_TEMPLATE2(linear::BarycentricMapperTetrahedronSetTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    typedef linear::BarycentricMapperTetrahedronSetTopology<TIn, TOut> Inherit;
    typedef typename linear::BarycentricMapper<TIn, TOut>::MappingData3D MappingData;

    PersistentContactBarycentricMapperTetrahedronSetTopology(
        topology::container::dynamic::TetrahedronSetTopologyContainer* fromTopology,
        topology::container::dynamic::PointSetTopologyContainer* toTopology)
        : linear::BarycentricMapperTetrahedronSetTopology<TIn, TOut>(fromTopology, toTopology)
    {
    }

    virtual ~PersistentContactBarycentricMapperTetrahedronSetTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::type::Vec3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);

    int keepContactPointFromInputMapping(const int /*index*/);

    void storeBarycentricData();

protected:

    sofa::type::vector< MappingData > m_storedMap;
};



template <class TIn, class TOut>
class PersistentContactBarycentricMapping : public linear::BarycentricMapping<TIn, TOut>, public PersistentContactMapping
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapping,TIn,TOut), SOFA_TEMPLATE2(linear::BarycentricMapping,TIn,TOut), PersistentContactMapping);

    typedef linear::BarycentricMapping<TIn, TOut> Inherit;
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

    typedef PersistentContactBarycentricMapper<InDataTypes,OutDataTypes> Mapper;

    PersistentContactBarycentricMapping()
        : Inherit()
        ,  m_persistentMapper(initLink("persistentMapper", "Internal persistent mapper created depending on the type of topology"))
        , m_init(false)
    {
    }

    PersistentContactBarycentricMapping(core::State<In>* from, core::State<Out>* to)
        : Inherit(from, to)
        , m_persistentMapper(initLink("persistentMapper", "Internal persistent mapper created depending on the type of topology"))
        , m_init(false)
    {
    }

    ~PersistentContactBarycentricMapping() override
    {
    }

    void init() override;

    void beginAddContactPoint() override;

    int addContactPointFromInputMapping(const sofa::type::Vec3& pos, std::vector< std::pair<int, double> > & baryCoords);

    int keepContactPointFromInputMapping(const int) override;

    void applyPositionAndFreePosition() override;

    void handleEvent(sofa::core::objectmodel::Event*) override;

protected:
    bool m_init;

    void createPersistentMapperFromTopology(BaseMeshTopology *topology);

    void storeBarycentricData();

    SingleLink<PersistentContactBarycentricMapping<In, Out>, Mapper, BaseLink::FLAG_STRONGLINK> m_persistentMapper;
};


using sofa::defaulttype::Vec3dTypes;


#if  !defined(SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_CPP)
extern template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3Types, Vec3Types >;


#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_H
