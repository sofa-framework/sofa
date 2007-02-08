//
// C++ Interface: AnimateBeginEvent
//
// Description:
//
//
// Author: Jeremie Allard, MGH/CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_TREE_ANIMATEBEGINEVENT_H
#define SOFA_SIMULATION_TREE_ANIMATEBEGINEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

/**
  Event fired by Simulation::animate() before computing a new animation step.
  @author Jeremie Allard
*/
class AnimateBeginEvent : public sofa::core::objectmodel::Event
{
public:
    AnimateBeginEvent( double dt );

    ~AnimateBeginEvent();

    double getDt() const { return dt; }

protected:
    double dt;
};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
