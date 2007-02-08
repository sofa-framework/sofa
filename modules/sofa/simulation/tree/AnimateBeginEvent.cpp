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
#include <sofa/simulation/tree/AnimateBeginEvent.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

AnimateBeginEvent::AnimateBeginEvent(double dt)
    : sofa::core::objectmodel::Event()
    , dt(dt)
{
}


AnimateBeginEvent::~AnimateBeginEvent()
{
}

} // namespace tree

} // namespace simulation

} // namespace sofa
