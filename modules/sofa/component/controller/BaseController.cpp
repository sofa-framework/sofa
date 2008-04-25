/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Implementation : Controller
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <sofa/component/controller/BaseController.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/JoystickEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

#include <sofa/simulation/tree/AnimateBeginEvent.h>
#include <sofa/simulation/tree/AnimateEndEvent.h>

#include <iostream>


namespace sofa
{

namespace component
{

namespace controller
{


BaseController::BaseController()
    : handleEventTriggersUpdate( initData(&handleEventTriggersUpdate, "handleEventTriggersUpdate", "Event handling frequency controls the controller update frequency" ) )
{

}



void BaseController::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::tree::AnimateBeginEvent *>(event))
    {
        onBeginAnimationStep();
    }
    else if (dynamic_cast<sofa::simulation::tree::AnimateEndEvent *>(event))
    {
        onEndAnimationStep();
    }
    else if (dynamic_cast<sofa::core::objectmodel::KeypressedEvent *>(event))
    {
        sofa::core::objectmodel::KeypressedEvent *kpev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent *>(event))
    {
        sofa::core::objectmodel::KeyreleasedEvent *krev = dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(krev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event))
    {
        sofa::core::objectmodel::MouseEvent *mev = dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event);
        onMouseEvent(mev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::JoystickEvent *>(event))
    {
        sofa::core::objectmodel::JoystickEvent *jev = dynamic_cast<sofa::core::objectmodel::JoystickEvent *>(event);
        onJoystickEvent(jev);
    }
}

} // namespace controller

} // namepace component

} // namespace sofa

