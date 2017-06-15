/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaBaseMechanics/BarycentricMapping.h>

#include "PersistentContactMapping.h"
#include <PersistentContact/config.h>

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
    virtual int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::defaulttype::Vector3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/) = 0;

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
class PersistentContactBarycentricMapperMeshTopology : public BarycentricMapperMeshTopology< TIn , TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperMeshTopology, TIn, TOut), SOFA_TEMPLATE2(BarycentricMapperMeshTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    PersistentContactBarycentricMapperMeshTopology(core::topology::BaseMeshTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology,
            helper::ParticleMask *_maskFrom,
            helper::ParticleMask *_maskTo)
        : BarycentricMapperMeshTopology<TIn, TOut>(fromTopology, toTopology, _maskFrom, _maskTo)
    {
    }

    virtual ~PersistentContactBarycentricMapperMeshTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::defaulttype::Vector3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);
};



template<class TIn, class TOut>
class PersistentContactBarycentricMapperSparseGridTopology : public BarycentricMapperSparseGridTopology< TIn , TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperSparseGridTopology, TIn, TOut), SOFA_TEMPLATE2(BarycentricMapperSparseGridTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    typedef BarycentricMapperSparseGridTopology< TIn , TOut> Inherit;
    typedef typename Inherit::CubeData CubeData;

    PersistentContactBarycentricMapperSparseGridTopology(topology::SparseGridTopology* fromTopology,
            topology::PointSetTopologyContainer* toTopology,
            helper::ParticleMask *_maskFrom,
            helper::ParticleMask *_maskTo)
        : BarycentricMapperSparseGridTopology<TIn, TOut>(fromTopology, toTopology, _maskFrom, _maskTo)
    {
    }

    virtual ~PersistentContactBarycentricMapperSparseGridTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::defaulttype::Vector3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);

    int keepContactPointFromInputMapping(const int /*index*/);

    void storeBarycentricData();

protected:
    sofa::helper::vector< CubeData > m_storedMap;
};



template<class TIn, class TOut>
class PersistentContactBarycentricMapperTetrahedronSetTopology : public BarycentricMapperTetrahedronSetTopology< TIn , TOut>, public PersistentContactBarycentricMapper< TIn , TOut>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapperTetrahedronSetTopology, TIn, TOut), SOFA_TEMPLATE2(BarycentricMapperTetrahedronSetTopology, TIn, TOut), SOFA_TEMPLATE2(PersistentContactBarycentricMapper, TIn, TOut));

    typedef TIn In;
    typedef TOut Out;
    typedef In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef typename InDataTypes::Real Real;

    typedef BarycentricMapperTetrahedronSetTopology<TIn, TOut> Inherit;
    typedef typename Inherit::MappingData MappingData;

    PersistentContactBarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology,
            topology::PointSetTopologyContainer* toTopology,
            helper::ParticleMask *_maskFrom,
            helper::ParticleMask *_maskTo)
        : BarycentricMapperTetrahedronSetTopology<TIn, TOut>(fromTopology, toTopology, _maskFrom, _maskTo)
    {
    }

    virtual ~PersistentContactBarycentricMapperTetrahedronSetTopology()
    {
    }

    int addContactPointFromInputMapping(const InVecDeriv& /*in*/, const sofa::defaulttype::Vector3& /*pos*/, std::vector< std::pair<int, double> > & /*baryCoords*/);

    int keepContactPointFromInputMapping(const int /*index*/);

    void storeBarycentricData();

protected:

    sofa::helper::vector< MappingData > m_storedMap;
};



template <class TIn, class TOut>
class PersistentContactBarycentricMapping : public BarycentricMapping<TIn, TOut>, public PersistentContactMapping
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(PersistentContactBarycentricMapping,TIn,TOut), SOFA_TEMPLATE2(BarycentricMapping,TIn,TOut), PersistentContactMapping);

    typedef BarycentricMapping<TIn, TOut> Inherit;
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
    {
    }

    PersistentContactBarycentricMapping(core::State<In>* from, core::State<Out>* to)
        : Inherit(from, to)
        , m_persistentMapper(initLink("persistentMapper", "Internal persistent mapper created depending on the type of topology"))
    {
    }

    virtual ~PersistentContactBarycentricMapping()
    {
    }

    virtual void init();

    void beginAddContactPoint();

    int addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & baryCoords);

    int keepContactPointFromInputMapping(const int);

    void applyPositionAndFreePosition();

    void handleEvent(sofa::core::objectmodel::Event*);

protected:
    bool m_init;

    void createPersistentMapperFromTopology(BaseMeshTopology *topology);

    void storeBarycentricData();

    SingleLink<PersistentContactBarycentricMapping<In, Out>, Mapper, BaseLink::FLAG_STRONGLINK> m_persistentMapper;
};


#ifndef SOFA_FLOAT
using sofa::defaulttype::Vec3dTypes;
#endif
#ifndef SOFA_DOUBLE
using sofa::defaulttype::Vec3fTypes;
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3fTypes, Vec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_PERSISTENTCONTACT_API PersistentContactBarycentricMapping< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_PERSISTENTCONTACTBARYCENTRICMAPPING_H
