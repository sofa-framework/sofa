/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_H
#define SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_H

#include <sofa/component/mapping/BarycentricMapping.h>

#include "ContinuousContactMapping.h"
#include "ContinuousContact.h"

namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class ContinuousContactBarycentricMapping : public BarycentricMapping<TIn, TOut>, public ContinuousContactMapping
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(ContinuousContactBarycentricMapping,TIn,TOut), SOFA_TEMPLATE2(BarycentricMapping,TIn,TOut), ContinuousContactMapping);

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

    typedef TopologyBarycentricMapper<InDataTypes,OutDataTypes> Mapper;
    typedef BarycentricMapperRegularGridTopology<InDataTypes, OutDataTypes> RegularGridMapper;
    typedef BarycentricMapperHexahedronSetTopology<InDataTypes, OutDataTypes> HexaMapper;

    ContinuousContactBarycentricMapping(core::State<In>* from, core::State<Out>* to)
        : Inherit(from, to)
    {
    }

    virtual ~ContinuousContactBarycentricMapping()
    {
    }

    virtual void init();

    void beginAddContactPoint();

    int addContactPointFromInputMapping(const sofa::defaulttype::Vector3& pos, std::vector< std::pair<int, double> > & baryCoords);

    bool m_init;
};


using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousContactBarycentricMapping< Vec3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousContactBarycentricMapping< Vec3fTypes, Vec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousContactBarycentricMapping< Vec3dTypes, Vec3fTypes >;
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousContactBarycentricMapping< Vec3fTypes, Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_CONTINUOUSCONTACTBARYCENTRICMAPPING_H
