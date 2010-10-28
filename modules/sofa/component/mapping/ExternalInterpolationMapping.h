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
#ifndef SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_H
#define SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_H

#include <sofa/core/Mapping.inl>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/vector.h>


namespace sofa
{

namespace component
{

namespace mapping
{


/**
 * @class ExternalInterpolationMapping
 * @brief Compute the mapping of points based on a given interpolation table
 */
template <class TIn, class TOut>
class ExternalInterpolationMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(ExternalInterpolationMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename std::pair<unsigned int, Real> couple;
    //typedef typename  InterpolationValueTable;

    Data< sofa::helper::vector<sofa::helper::vector< unsigned int > > > f_interpolationIndices;
    Data< sofa::helper::vector<sofa::helper::vector< Real > > > f_interpolationValues;

    ExternalInterpolationMapping(core::State<In>* from, core::State<Out>* to);

    void clear(int /*reserve*/) {}

    int addPoint(int /*index*/) {return 0;}

    void init();

    // handle topology changes depending on the topology
    void handleTopologyChange(core::topology::Topology* t);

    virtual ~ExternalInterpolationMapping();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );
private:
    bool doNotMap;
};

using sofa::defaulttype::Vec1dTypes;
using sofa::defaulttype::Vec2dTypes;
using sofa::defaulttype::Vec3dTypes;
using sofa::defaulttype::Vec1fTypes;
using sofa::defaulttype::Vec2fTypes;
using sofa::defaulttype::Vec3fTypes;
using sofa::defaulttype::ExtVec2fTypes;
using sofa::defaulttype::ExtVec3fTypes;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class ExternalInterpolationMapping< Vec3dTypes, Vec3dTypes >;
extern template class ExternalInterpolationMapping< Vec2dTypes, Vec2dTypes >;
extern template class ExternalInterpolationMapping< Vec1dTypes, Vec1dTypes >;
extern template class ExternalInterpolationMapping< Vec3dTypes, ExtVec3fTypes >;
extern template class ExternalInterpolationMapping< Vec2dTypes, ExtVec2fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class ExternalInterpolationMapping< Vec3fTypes, Vec3fTypes >;
extern template class ExternalInterpolationMapping< Vec2fTypes, Vec2fTypes >;
extern template class ExternalInterpolationMapping< Vec1fTypes, Vec1fTypes >;
extern template class ExternalInterpolationMapping< Vec3fTypes, ExtVec3fTypes >;
extern template class ExternalInterpolationMapping< Vec2fTypes, ExtVec2fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class ExternalInterpolationMapping< Vec3dTypes, Vec3fTypes >;
extern template class ExternalInterpolationMapping< Vec3fTypes, Vec3dTypes >;
extern template class ExternalInterpolationMapping< Vec2dTypes, Vec2fTypes >;
extern template class ExternalInterpolationMapping< Vec2fTypes, Vec2dTypes >;
extern template class ExternalInterpolationMapping< Vec1dTypes, Vec1fTypes >;
extern template class ExternalInterpolationMapping< Vec1fTypes, Vec1dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
