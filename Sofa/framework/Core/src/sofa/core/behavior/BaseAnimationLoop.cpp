/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/objectmodel/BaseNode.h>

#include <cstdlib>
#include <cmath>

namespace sofa::core::behavior
{

BaseAnimationLoop::BaseAnimationLoop()
{}

BaseAnimationLoop::~BaseAnimationLoop()
{}

void BaseAnimationLoop::init()
{
    Inherit1::init();

    if(!l_node)
        l_node = dynamic_cast<sofa::core::objectmodel::BaseNode*>(getContext());
}
void BaseAnimationLoop::integrate(const core::ExecParams* params, SReal dt)
{
    step(params, dt);
}


} // namespace sofa::core::behavior

