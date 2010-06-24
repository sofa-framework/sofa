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

#include <sofa/core/behavior/MechanicalMapping.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <class BasicMapping>
class CurveMapping : public BasicMapping
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CurveMapping,BasicMapping), BasicMapping);
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::DataTypes DataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename std::map<unsigned int, Deriv>::const_iterator OutConstraintIterator;
    typedef typename Out::Real Real;

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

    CurveMapping(In* from, Out* to)
        : Inherit(from, to),
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

    void init();
    void reinit();
    void storeResetState();
    void reset();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    void handleEvent(sofa::core::objectmodel::Event* event);

    void draw();

    Real advanceAbscissa(Real ab, Real dist);
    void rotateElements();
};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
