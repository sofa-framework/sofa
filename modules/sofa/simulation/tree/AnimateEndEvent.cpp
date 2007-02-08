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
#include <sofa/simulation/tree/AnimateEndEvent.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

AnimateEndEvent::AnimateEndEvent(double dt)
    : sofa::core::objectmodel::Event()
    , dt(dt)
{
}


AnimateEndEvent::~AnimateEndEvent()
{
}

} // namespace tree

} // namespace simulation

} // namespace sofa
