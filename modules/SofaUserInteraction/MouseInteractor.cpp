/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_COLLISION_MOUSEINTERACTOR_CPP
#include <SofaUserInteraction/MouseInteractor.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(MouseInteractor)

int MouseInteractorClass = core::RegisterObject("Perform tasks related to the interaction with the mouse")
#ifndef SOFA_DOUBLE
        .add< MouseInteractor<defaulttype::Vec2fTypes> >()
        .add< MouseInteractor<defaulttype::Vec3fTypes> >()
#endif
#ifndef SOFA_FLOAT
        .add< MouseInteractor<defaulttype::Vec2dTypes> >()
        .add< MouseInteractor<defaulttype::Vec3dTypes> >()
#endif
        ;
int MouseInteractorRigidClass = core::RegisterObject("Perform tasks related to the interaction with the mouse and rigid objects")
#ifndef SOFA_DOUBLE
        .add< MouseInteractor<defaulttype::Rigid3fTypes> >()
#endif
#ifndef SOFA_FLOAT
        .add< MouseInteractor<defaulttype::Rigid3dTypes> >()
#endif
        ;

#ifndef SOFA_DOUBLE
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Vec2fTypes>;
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Vec3fTypes>;
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Vec2dTypes>;
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Vec3dTypes>;
template class SOFA_USER_INTERACTION_API MouseInteractor<defaulttype::Rigid3dTypes>;
#endif


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
    VecPerformer::iterator found=std::find(performers.begin(), performers.end(), i);
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
    VecPerformer::iterator it=performers.begin(), it_end=performers.end();
    for (; it!=it_end; ++it)
    {
        (*it)->execute();
    }
}



void BaseMouseInteractor::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    VecPerformer::iterator it=performers.begin(), it_end=performers.end();
    for (; it!=it_end; ++it)
        (*it)->draw(vparams);

    if( !vparams->isSupported(sofa::core::visual::API_OpenGL) ) return;

    if (lastPicked.body)
    {
        if (isAttached)
            glColor4f(1.0f,0.0f,0.0f,1.0f);
        else
            glColor4f(0.0f,1.0f,0.0f,1.0f);

        glDisable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3);
        lastPicked.body->draw(vparams,lastPicked.indexCollisionElement);

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


        glColor4f(1,1,1,1);
        glLineWidth(1);
    }
#endif /* SOFA_NO_OPENGL */
}
}
}
}
