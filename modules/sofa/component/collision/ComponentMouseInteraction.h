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
#ifndef SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_H
#define SOFA_COMPONENT_COLLISION_COMPONENTMOUSEINTERACTION_H

#include <sofa/simulation/common/Simulation.h>
/* #include <sofa/simulation/common/InitVisitor.h> */
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/component/mapping/IdentityMapping.h>

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{

using simulation::Node;
using sofa::component::collision::BodyPicked;

class ComponentMouseInteraction
{
public:
    ComponentMouseInteraction():parentNode(NULL), nodeRayPick(NULL)/* ,mouseCollision(NULL) */
    {
    }

    virtual ~ComponentMouseInteraction()
    {
        if (!nodeRayPick) return;
        nodeRayPick->execute<simulation::DeleteVisitor>();
        delete nodeRayPick;
    }

    virtual void init(Node* node)
    {
        parentNode = node;
        nodeRayPick = simulation::getSimulation()->newNode("RayPick");
    }

    void activate()
    {
        parentNode->addChild(nodeRayPick);
    }

    void deactivate()
    {
        mouseInteractor->doReleaseBody();
        nodeRayPick->detachFromGraph();
    }

    void reset()
    {
        mouseInteractor->reset();
    }

    virtual bool isCompatible( core::objectmodel::BaseContext *)const=0;

    typedef helper::Factory<std::string, ComponentMouseInteraction, core::objectmodel::BaseContext*> ComponentMouseInteractionFactory;

    template <class RealObject>
    static void create( RealObject*& obj, core::objectmodel::BaseContext* /* context */)
    {
        obj = new RealObject;
    }


    //Components
    Node                                                      *parentNode;
    Node                                                      *nodeRayPick;
    sofa::core::componentmodel::behavior::BaseMechanicalState *mouseInSofa;
    sofa::core::componentmodel::behavior::BaseMechanicalMapping *mouseMapping;
    sofa::component::collision::BaseMouseInteractor           *mouseInteractor;
};



template <class DataTypes>
class TComponentMouseInteraction : public ComponentMouseInteraction
{
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MousePosition;
    typedef sofa::component::container::MechanicalObject< DataTypes > MouseContainer;
    typedef sofa::component::collision::MouseInteractor< DataTypes > Interactor;
    typedef sofa::component::mapping::IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping<sofa::core::componentmodel::behavior::MechanicalState< defaulttype::Vec3Types>, sofa::core::componentmodel::behavior::MechanicalState< DataTypes > > > IdentityMechanicalMapping;

public:


    void init(Node* node)
    {
        ComponentMouseInteraction::init(node);

        parentNode->addChild(nodeRayPick);

        mouseInSofa =  new MouseContainer;
        mouseInSofa = new MouseContainer; mouseInSofa->resize(1);
        mouseInSofa->setName("MousePosition");
        nodeRayPick->addObject(mouseInSofa);



        mouseInteractor = new Interactor;
        mouseInteractor->setName("MouseInteractor");
        nodeRayPick->addObject(mouseInteractor);

        MousePosition *mecha = dynamic_cast<MousePosition*>(node->getMechanicalState());
        mouseMapping = new IdentityMechanicalMapping(mecha, static_cast<MouseContainer*>(mouseInSofa));
        nodeRayPick->addObject(mouseMapping);

        mouseMapping->setMechanical(false);

        nodeRayPick->init();
        mouseInSofa->init();
        mouseMapping->init();
        mouseInteractor->init();

        /*             parentNode->execute<simulation::InitVisitor>(); */
        parentNode->removeChild(nodeRayPick);
    }



    bool isCompatible( core::objectmodel::BaseContext *context) const
    {
        return dynamic_cast<MouseContainer*>(context->getMechanicalState());
    }

};
}
}
}

#endif
