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

#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <vector>
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
    typedef BasicMapping Inherit;
    typedef typename Inherit::In In;
    typedef typename Inherit::Out Out;
    typedef typename Out::VecCoord VecCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::SparseDeriv OutSparseDeriv;
    typedef typename In::Coord InCoord;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::Real Real;

    Data < helper::vector<Real> > abscissa;
    Data < helper::vector<Real> > angle;
    Data < Real > step;
    Data < Real > angularStep;
    Data < Real > numNodes;
    Data < Real > stepNode;
    Data < Real > velocity;

    helper::vector<int> old_integer;
    helper::vector<double> old_angle;

    CurveMapping(In* from, Out* to)
        : Inherit(from, to),
          abscissa( initData(&abscissa, "abscissa", "")),
          angle( initData(&angle, "angle", "")),
          step( initData(&step, (Real) 10.0, "step", "")),
          angularStep( initData(&angularStep, (Real) 0.0, "angularStep", "")),
          numNodes( initData(&numNodes,  5.0, "numNodes", "")),
          stepNode( initData(&stepNode, (Real) 0.5, "stepNode", "")),
          velocity( initData(&velocity, (Real) 0.0, "velocity", ""))
    {
    }

    virtual ~CurveMapping()
    {
    }

    void init();

    void apply( typename Out::VecCoord& out, const typename In::VecCoord& in );

    void applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in );

    void applyJT( typename In::VecDeriv& out, const typename Out::VecDeriv& in );

    void applyJT( typename In::VecConst& out, const typename Out::VecConst& in );

    void handleEvent(sofa::core::objectmodel::Event* event);

    void draw();

};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
