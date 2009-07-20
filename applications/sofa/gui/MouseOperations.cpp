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
#include <sofa/gui/MouseOperations.h>
#include <sofa/gui/PickHandler.h>

namespace sofa
{

namespace gui
{

void AttachOperation::start()
{
    pickHandle->getInteraction()->mouseInteractor->doAttachBody(*pickHandle->getLastPicked(), getStiffness());
}

void AttachOperation::execution()
{
    //do nothing
}

void AttachOperation::end()
{
    pickHandle->getInteraction()->mouseInteractor->doReleaseBody();
}




void RemoveOperation::start()
{
    execution();
}

void RemoveOperation::execution()
{
    pickHandle->getInteraction()->mouseInteractor->doRemoveCollisionElement(*pickHandle->getLastPicked());
}

void RemoveOperation::end()
{
    //do nothing
}


void InciseOperation::start()
{
    execution();
}


void InciseOperation::execution()
{
    helper::fixed_array< BodyPicked,2 > &elementsPicked = *pickHandle->getElementsPicked();

    elementsPicked[1] = elementsPicked[0];
    elementsPicked[0] = *pickHandle->getLastPicked();
    pickHandle->getInteraction()->mouseInteractor->doInciseBody(elementsPicked);
}

void InciseOperation::end()
{
    pickHandle->getInteraction()->mouseInteractor->doInciseBody(*pickHandle->getElementsPicked());
}


void FixOperation::start()
{
    pickHandle->getInteraction()->mouseInteractor->doFixParticle(*pickHandle->getLastPicked(), getStiffness());
}

void FixOperation::execution()
{
}

void FixOperation::end()
{
    //do nothing
}


}
}
