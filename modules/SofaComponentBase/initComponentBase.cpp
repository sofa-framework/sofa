/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/system/config.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaBaseTopology/initBaseTopology.h>
#include <SofaBaseMechanics/initBaseMechanics.h>
#include <SofaBaseCollision/initBaseCollision.h>
#include <SofaBaseLinearSolver/initBaseLinearSolver.h>
#include <SofaBaseAnimationLoop/initBaseAnimationLoop.h>
#include <SofaBaseVisual/initBaseVisual.h>

#include "messageHandlerComponent.h"
using sofa::component::logging::MessageHandlerComponent ;
using sofa::component::logging::FileMessageHandlerComponent ;

namespace sofa
{

namespace component
{


void initComponentBase()
{
    static bool first = true;
    if (first)
    {
        initBaseTopology();
        initBaseMechanics();
        initBaseCollision();
        initBaseLinearSolver();
        initBaseAnimationLoop();
        initBaseVisual();
        first = false;
    }
}

SOFA_LINK_CLASS(MessageHandlerComponent)
SOFA_LINK_CLASS(FileMessageHandlerComponent)

} // namespace component

} // namespace sofa
