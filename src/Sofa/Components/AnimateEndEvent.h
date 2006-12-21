//
// C++ Interface: AnimateEndEvent
//
// Description:
//
//
// Author: Jeremie Allard, MGH/CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENTS_ANIMATEENDEVENT_H
#define SOFA_COMPONENTS_ANIMATEENDEVENT_H

#include <Sofa/Abstract/Event.h>

namespace Sofa
{

namespace Components
{

/**
  Event fired by Simulation::animate() after computing a new animation step.
  @author Jeremie Allard
*/
class AnimateEndEvent : public Sofa::Abstract::Event
{
public:
    AnimateEndEvent( double dt );

    ~AnimateEndEvent();

    double getDt() const { return dt; }

protected:
    double dt;
};

}

}

#endif
