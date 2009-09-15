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

namespace sofa
{

namespace component
{

namespace collision
{

class BaseMouseInteractor;
template <class DataTypes>
class MouseInteractor;


class SOFA_COMPONENT_COLLISION_API InteractionPerformer
{
public:
    typedef helper::Factory<std::string, InteractionPerformer, BaseMouseInteractor*> InteractionPerformerFactory;

    InteractionPerformer(BaseMouseInteractor *i):interactor(i) {};
    virtual ~InteractionPerformer() {};


    virtual void start()=0;
    virtual void execute()=0;


    virtual void handleEvent(core::objectmodel::Event * ) {};
    virtual void draw() {};

    template <class RealObject>
    static void create( RealObject*& obj, BaseMouseInteractor* interactor)
    {
        obj = new RealObject(interactor);
    }
    BaseMouseInteractor *interactor;
};


template <class DataTypes>
class TInteractionPerformer: public InteractionPerformer
{
public:

    TInteractionPerformer(BaseMouseInteractor *i):InteractionPerformer(i) {};

    template <class RealObject>
    static void create( RealObject*& obj, BaseMouseInteractor* interactor)
    {
        if (!dynamic_cast< MouseInteractor<DataTypes>* >(interactor)) obj=NULL;
        else obj = new RealObject(interactor);
    }

};
}
}
}

#endif
