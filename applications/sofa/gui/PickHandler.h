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
#ifndef SOFA_GUI_PICKHANDLER_H
#define SOFA_GUI_PICKHANDLER_H

#include <sofa/gui/OperationFactory.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/Node.h>

#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/ComponentMouseInteraction.h>

#include <sofa/helper/fixed_array.h>

namespace sofa
{

namespace gui
{

using simulation::Node;
using sofa::component::collision::BodyPicked;
using sofa::component::collision::ComponentMouseInteraction;



class PickHandler
{
    typedef sofa::component::collision::RayModel MouseCollisionModel;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MouseContainer;
public:


    PickHandler();
    ~PickHandler();

    void activateRay(bool act);

    void updateRay(const sofa::defaulttype::Vector3 &position, const sofa::defaulttype::Vector3 &orientation);

    void handleMouseEvent( MOUSE_STATUS status, MOUSE_BUTTON button);

    void init();
    void reset();

    void changeOperation(MOUSE_BUTTON button, const std::string &op)
    {
        if (operations[button]) delete operations[button];
        operations[button] = OperationFactory::Instanciate(op);
        operations[button]->configure(this,button);
    }

    ComponentMouseInteraction           *getInteraction();
    BodyPicked                          *getLastPicked() {return &lastPicked;};

protected:

    Node                *mouseNode;
    MouseContainer      *mouseContainer;
    MouseCollisionModel *mouseCollision;


    BodyPicked findCollision();
    BodyPicked findCollisionUsingPipeline();
    BodyPicked findCollisionUsingBruteForce();
    bool needToCastRay();
    void setCompatibleInteractor();

    ComponentMouseInteraction *interaction;
    std::vector< ComponentMouseInteraction *> instanceComponents;

    bool interactorInUse;

    BodyPicked lastPicked;

    MOUSE_BUTTON mouseButton;
    MOUSE_STATUS mouseStatus;

    helper::fixed_array< Operation*,3 > operations;
    bool useCollisions;
};
}
}

#endif
