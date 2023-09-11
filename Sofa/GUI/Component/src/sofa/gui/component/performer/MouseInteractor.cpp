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
#define SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_CPP
#include <sofa/gui/component/performer/MouseInteractor.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gui::component::performer
{

int MouseInteractorClass = core::RegisterObject("Perform tasks related to the interaction with the mouse")
        .add< MouseInteractor<defaulttype::Vec2Types> >()
        .add< MouseInteractor<defaulttype::Vec3Types> >()

        ;
int MouseInteractorRigidClass = core::RegisterObject("Perform tasks related to the interaction with the mouse and rigid objects")
        .add< MouseInteractor<defaulttype::Rigid3Types> >()

        ;

template class SOFA_GUI_COMPONENT_API MouseInteractor<defaulttype::Vec2Types>;
template class SOFA_GUI_COMPONENT_API MouseInteractor<defaulttype::Vec3Types>;
template class SOFA_GUI_COMPONENT_API MouseInteractor<defaulttype::Rigid3Types>;



void BaseMouseInteractor::cleanup()
{
    while (!performers.empty())
    {
        removeInteractionPerformer(*performers.begin());
    }
    lastPicked=BodyPicked();
}


void BaseMouseInteractor::handleEvent(core::objectmodel::Event *e)
{
    VecPerformer::iterator it=performers.begin(), it_end=performers.end();
    for (; it!=it_end; ++it)
    {
        (*it)->handleEvent(e);
    }
}

void BaseMouseInteractor::addInteractionPerformer( InteractionPerformer *perf)
{
    performers.insert(performers.end(),perf);
}

bool BaseMouseInteractor::removeInteractionPerformer( InteractionPerformer *i)
{
    const VecPerformer::iterator found=std::find(performers.begin(), performers.end(), i);
    if (found == performers.end()) return false;
    else
    {
//            delete *found; //Only remove the Performer from the Interactor, do not delete it!
        performers.erase(found);
        return true;
    }
}

void BaseMouseInteractor::updatePosition(SReal )
{
    for (const auto perf : performers)
    {
        perf->execute();
    }
}



void BaseMouseInteractor::draw(const core::visual::VisualParams* vparams)
{
    VecPerformer::iterator it=performers.begin(), it_end=performers.end();
    for (; it!=it_end; ++it)
        (*it)->draw(vparams);

    if (lastPicked.body)
    {
        vparams->drawTool()->setPolygonMode(0, true);
        lastPicked.body->draw(vparams,lastPicked.indexCollisionElement);
        vparams->drawTool()->setPolygonMode(0, false);

    }
}

} // namespace sofa::gui::component::performer
