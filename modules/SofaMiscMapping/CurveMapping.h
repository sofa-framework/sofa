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
//
// C++ Interface: CurveMapping
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_MAPPING_CURVEMAPPING_H
#define SOFA_COMPONENT_MAPPING_CURVEMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>

#include <sofa/defaulttype/Quat.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>



namespace sofa
{
namespace core
{
namespace objectmodel
{
class Event;
} // namespace objectmodel
} // namespace core
} // namespace sofa


namespace sofa
{

namespace component
{

namespace mapping
{

template <class TIn, class TOut>
class CurveMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CurveMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out DataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::Real Real;

    typedef Data< typename Out::VecCoord > OutDataVecCoord;
    typedef Data< typename In::VecCoord > InDataVecCoord;
    typedef Data< typename Out::VecDeriv > OutDataVecDeriv;
    typedef Data< typename In::VecDeriv > InDataVecDeriv;

    Data < helper::vector<Real> > abscissa;
    Data < helper::vector<Real> > angle;
    Data < Real > step;
    Data < Real > angularStep;
    Data < int > numNodes;
    Data < Real > stepNode;
    Data < Real > distNode;
    Data < Real > velocity;

    helper::vector<int> old_integer;
    helper::vector<double> old_angle;
    helper::vector<Real> lengthElements;
    helper::vector<defaulttype::Quat> quatElements;
    helper::vector<defaulttype::Quat> rotatedQuatElements;
    helper::vector<defaulttype::Quat> quatInitNodes;

    helper::vector<Real> reset_abscissa;

    void init();
    void reinit();
    void storeResetState();
    void reset();

    void apply(const core::MechanicalParams* mparams, OutDataVecCoord& out, const InDataVecCoord& in);

    void applyJ(const core::MechanicalParams* mparams, OutDataVecDeriv& out, const InDataVecDeriv& in);

    void applyJT(const core::MechanicalParams* mparams, InDataVecDeriv& out, const OutDataVecDeriv& in);

    void applyJT(const core::ConstraintParams *cparams, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in);

    void handleEvent(sofa::core::objectmodel::Event* event);

    void draw(const core::visual::VisualParams* vparams);

    Real advanceAbscissa(Real ab, Real dist);
    void rotateElements();

protected:

    CurveMapping()
        : Inherit(),
          abscissa( initData(&abscissa, "abscissa", "")),
          angle( initData(&angle, "angle", "")),
          step( initData(&step, (Real) 10.0, "step", "")),
          angularStep( initData(&angularStep, (Real) 0.0, "angularStep", "")),
          numNodes( initData(&numNodes,  (int)5, "numNodes", "")),
          stepNode( initData(&stepNode, (Real) 0.0, "stepNode", "")),
          distNode( initData(&distNode, (Real) 0.0, "distNode", "")),
          velocity( initData(&velocity, (Real) 0.0, "velocity", ""))
    {
    }

    virtual ~CurveMapping()
    {
    }
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_CURVEMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API CurveMapping< defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CurveMapping< defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CurveMapping< defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes >;
extern template class SOFA_MISC_MAPPING_API CurveMapping< defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MAPPING_CURVEMAPPING_H
