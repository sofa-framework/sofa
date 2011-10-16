/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_INTERACTIONPERFORMER_H
#define SOFA_COMPONENT_COLLISION_INTERACTIONPERFORMER_H

#include <sofa/component/component.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace collision
{

class BaseMouseInteractor;
template <class DataTypes>
class MouseInteractor;


class SOFA_USER_INTERACTION_API InteractionPerformer
{
public:
    typedef helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*> InteractionPerformerFactory;

    InteractionPerformer(BaseMouseInteractor *i):interactor(i),freezePerformer(0) {}
    virtual ~InteractionPerformer() {}

    virtual void start()=0;
    virtual void execute()=0;

    virtual void handleEvent(core::objectmodel::Event * ) {}
    virtual void draw(const core::visual::VisualParams* ) {}

    virtual void setPerformerFreeze() {freezePerformer = true;}

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        return new RealObject(interactor);
    }
    BaseMouseInteractor *interactor;
    bool freezePerformer;
};


template <class DataTypes>
class TInteractionPerformer: public InteractionPerformer
{
public:

    TInteractionPerformer(BaseMouseInteractor *i):InteractionPerformer(i) {};

    template <class RealObject>
    static RealObject* create( RealObject*, BaseMouseInteractor* interactor)
    {
        if (!dynamic_cast< MouseInteractor<DataTypes>* >(interactor)) return NULL;
        else return new RealObject(interactor);
    }

};
#if defined WIN32 && !defined(SOFA_COMPONENT_COLLISION_INTERACTIONPERFOMER_CPP)
//delay load of the specialized Factory class. unique definition reside in the cpp file.
extern template class SOFA_USER_INTERACTION_API helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*>;
#endif

}
}
}

#endif
