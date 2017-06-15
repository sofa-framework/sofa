/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/helper/system/config.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaBaseTopology/initBaseTopology.h>
#include <SofaBaseMechanics/initBaseMechanics.h>
#include <SofaBaseCollision/initBaseCollision.h>
#include <SofaBaseLinearSolver/initBaseLinearSolver.h>
#include <SofaBaseVisual/initBaseVisual.h>

#include "messageHandlerComponent.h"
using sofa::component::logging::MessageHandlerComponent ;
using sofa::component::logging::FileMessageHandlerComponent ;

#include "MakeAliasComponent.h"
using sofa::component::MakeAliasComponent ;

#include "MakeDataAliasComponent.h"
using sofa::component::MakeAliasComponent ;

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
        initBaseVisual();
        first = false;
    }
}

SOFA_LINK_CLASS(MakeAliasComponent)
SOFA_LINK_CLASS(MakeDataAliasComponent)
SOFA_LINK_CLASS(MessageHandlerComponent)
SOFA_LINK_CLASS(FileMessageHandlerComponent)

} // namespace component

} // namespace sofa
