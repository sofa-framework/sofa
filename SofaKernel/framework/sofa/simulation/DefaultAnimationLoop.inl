/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_SIMULATION_DEFAULTANIMATIONLOOP_INL
#define SOFA_SIMULATION_DEFAULTANIMATIONLOOP_INL

#include <sofa/simulation/Node.h>
#include "DefaultAnimationLoop.h"

namespace sofa
{
namespace simulation
{

template<class T>
typename T::SPtr DefaultAnimationLoop::create(T*, BaseContext* context, BaseObjectDescription* arg)
{
    simulation::Node* gnode = dynamic_cast<simulation::Node*>(context);
    typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
    if (context) context->addObject(obj);
    if (arg) obj->parse(arg);
    return obj;
}

} // namespace simulation
} // namespace sofa

#endif  /* SOFA_SIMULATION_DEFAULTANIMATIONLOOP_INL */
