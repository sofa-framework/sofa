//
// C++ Implementation: AnimateBeginEvent
//
// Description:
//
//
// Author: Jeremie Allard, MGH/CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "AnimateBeginEvent.h"

namespace Sofa
{

namespace Components
{

AnimateBeginEvent::AnimateBeginEvent(double dt)
    : Sofa::Abstract::Event()
    , dt(dt)
{
}


AnimateBeginEvent::~AnimateBeginEvent()
{
}

}

}
