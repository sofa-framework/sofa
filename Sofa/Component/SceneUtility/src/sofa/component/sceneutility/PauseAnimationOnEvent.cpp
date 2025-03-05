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
#include <sofa/component/sceneutility/PauseAnimationOnEvent.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/PauseEvent.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::sceneutility
{

PauseAnimationOnEvent::PauseAnimationOnEvent() : paused(false)
{
}


PauseAnimationOnEvent::~PauseAnimationOnEvent()
{

}

void PauseAnimationOnEvent::init()
{
    PauseAnimation::init();
    this->f_listening.setValue(true);
}

bool PauseAnimationOnEvent::isPaused()
{
    return paused;
}

void PauseAnimationOnEvent::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (sofa::simulation::PauseEvent::checkEventType(event))
    {
        paused = true;
        pause();
    }
}

void registerPauseAnimationOnEvent(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("This component pauses the simulation when receiving a PauseEvent.")
        .add< PauseAnimationOnEvent >());
}

} // namespace sofa::component::sceneutility
