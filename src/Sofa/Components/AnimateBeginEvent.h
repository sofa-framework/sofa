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
#ifndef SOFA_COMPONENTS_ANIMATEBEGINEVENT_H
#define SOFA_COMPONENTS_ANIMATEBEGINEVENT_H

#include <Sofa/Abstract/Event.h>

namespace Sofa
{

namespace Components
{

/**
  Event fired by Simulation::animate() before computing a new animation step.
  @author Jeremie Allard
*/
class AnimateBeginEvent : public Sofa::Abstract::Event
{
public:
    AnimateBeginEvent( double dt );

    ~AnimateBeginEvent();

    double getDt() const { return dt; }

protected:
    double dt;
};

}

}

#endif
