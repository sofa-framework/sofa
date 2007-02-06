//
// C++ Implementation: AnimateEndEvent
//
// Description:
//
//
// Author: Jeremie Allard, MGH/CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "AnimateEndEvent.h"

namespace Sofa
{

namespace Components
{

AnimateEndEvent::AnimateEndEvent(double dt)
    : Sofa::Abstract::Event()
    , dt(dt)
{
}


AnimateEndEvent::~AnimateEndEvent()
{
}

}

}
