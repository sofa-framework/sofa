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
#ifndef SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_H
#define SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>

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
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real OutReal;

    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Real InReal;
    typedef typename std::pair<unsigned int, Real> couple;
    //typedef typename  InterpolationValueTable;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    Data< sofa::helper::vector<sofa::helper::vector< unsigned int > > > f_interpolationIndices;
    Data< sofa::helper::vector<sofa::helper::vector< Real > > > f_interpolationValues;

    void clear(int /*reserve*/) {}

    int addPoint(int /*index*/) {return 0;}

    void init();

    // handle topology changes depending on the topology
    void handleTopologyChange(core::topology::Topology* t);

    virtual void apply( const sofa::core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in);
    //void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    virtual void applyJ( const sofa::core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in);
    //void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    virtual void applyJT( const sofa::core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in);
    //void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    virtual void applyJT( const sofa::core::ConstraintParams* cparams, InDataMatrixDeriv& out, const OutDataMatrixDeriv& in);
    //void applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in );


protected:
    ExternalInterpolationMapping();

    virtual ~ExternalInterpolationMapping();

private:
    bool doNotMap;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_EXTERNALINTERPOLATIONMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::ExtVec2fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::ExtVec2fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_MISC_MAPPING_API ExternalInterpolationMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
